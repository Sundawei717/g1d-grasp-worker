import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch

from chunk_model import SharedResNet18ChunkTransformer, masked_chunk_loss
from step1_episode import AlignedEpisode
from three_view_detector import write_json
from three_view_inputs import proprioception
from three_view_train import ThreeViewWindows, read_boxes, supervised_indices, supervision
from unified_grasp import LiftGateConfig, execute
from v3_detector import cache_key, detector_config, prompt_spec, save_episode_cache


class V3FinalTest(TestCase):
    def test_visual_shapes_zero_init_weights_and_batch_norm(self):
        torch.manual_seed(288)
        a = SharedResNet18ChunkTransformer(fixture_random_init=True, wrist_channels=3, v3_final=True, freeze_bn=True)
        torch.manual_seed(288)
        b = SharedResNet18ChunkTransformer(fixture_random_init=True, wrist_channels=4, v3_final=True, freeze_bn=True)
        self.assertEqual([a.head_stem.in_channels, a.left_wrist_stem.in_channels, a.right_wrist_stem.in_channels], [5,3,3])
        self.assertEqual([b.head_stem.in_channels, b.left_wrist_stem.in_channels, b.right_wrist_stem.in_channels], [5,3,4])
        self.assertEqual(tuple(b.position_grid.shape),(14,14,256))
        self.assertEqual(tuple(b.view_embedding.shape),(3,256))
        self.assertEqual(torch.count_nonzero(b.head_stem.weight[:,3:]).item(),0)
        self.assertEqual(torch.count_nonzero(b.right_wrist_stem.weight[:,3:]).item(),0)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for name, value in a.state_dict().items():
            other = b.state_dict()[name]
            torch.testing.assert_close(value, other[:, :3] if name=="right_wrist_stem.weight" else other)
        self.assertTrue(a.training)
        self.assertTrue(all(not module.training for module in a.backbone.modules() if isinstance(module,torch.nn.BatchNorm2d)))
        self.assertTrue(a.backbone[0].weight.requires_grad)
        head,left,right = (torch.rand(1,c,224,224) for c in (5,3,3))
        body,prompt=torch.zeros(1,2,17),torch.zeros(1,6)
        b_right=torch.cat((right,torch.ones(1,1,224,224)),dim=1)
        with torch.inference_mode():
            a.eval(); b.eval()
            self.assertEqual(tuple(b.visual_tokens(head,left,b_right).shape),(1,588,256))
            out_a,out_b=a(head,left,body,prompt,right),b(head,left,body,prompt,b_right)
            self.assertEqual(tuple(out_b.shape),(1,8,8))
            torch.testing.assert_close(out_a,out_b,rtol=2e-5,atol=2e-6)

    def test_executor_history_uses_every_applied_twentieth_second(self):
        tick = 0
        seen = []
        def frame(): return None,{"sensor_sim_time_s":tick/20}
        def feedback(): return {"sim_time_s":tick/20,"eef_xyz_m":[0.,0.,0.],"width_m":.085,
                                "pad_normal_force_n":{},"max_joint_speed_rad_s":.05}
        def source(step, _, __, state):
            seen.append((step,round(state["previous_control"]["sim_time_s"]*20)))
            return [.09,.09],{}
        def apply(width):
            nonlocal tick
            tick+=1
            return width,0.
        def unused(_): self.fail("Open jaws should never trigger")
        result=execute(initial_frame=frame(),capture=frame,source=source,feedback=feedback,
                       apply=apply,lift=unused,config=LiftGateConfig())
        self.assertEqual(seen[:4],[(0,0),(2,1),(4,3),(6,5)])
        self.assertEqual((result["status"],result["steps"]),("timeout",300))

    def test_lossless_dataset_right_only_cache_and_complete_close_supervision(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            path=root/"sample.h5"
            resolution={"prim_path":"/test/Camera","resolution":[224,224]}
            writer=AlignedEpisode({"case_id":"C_7000"},[f"joint_{i}" for i in range(7)],
                                  dict.fromkeys(("head_left","left_wrist","right_wrist"), resolution),
                                  {"frame_id":"0/20","bbox_0":[.1,.1,.9,.9],"affordance_0":[.5,.5],"target_query":"mug"})
            rgb=np.full((224,224,3),32,dtype=np.uint8)
            phases=("approach","approach","close","close","lift","hold")
            for i in range(7):
                stamps=dict(sensor_sim_time_s=i/20,reference_time=dict(numerator=i,denominator=20))
                frame=rgb,{**stamps,"wrist_views":{name:(rgb,stamps) for name in ("left_wrist","right_wrist")}}
                writer.observe(frame,[i*.1]*7,.09-i*.01,[i*.01,.1,.2,0.,0.,0.,1.])
                if i<len(phases):
                    writer.action(phases[i],np.tile([.3+i*.01]*7+[-.01,-.01],(30,1)))
            row={**writer.save(path,{"success":True}),"split":"train","group":"C_start","asset_family":"mug/00007"}
            manifest={"phase":"three_view_v3_split","source_sha256":"source","episodes":[row]}
            config=detector_config()
            class Detector:
                def spec(self, query): return prompt_spec(config,query)
            frames=[dict(frame_id=f"{i}/20",control_step_id=i,bbox=None if i!=1 else [.1,.1,.8,.8],
                         score=.8 if i==1 else None,missing=i!=1) for i in range(7)]
            entry=save_episode_cache(root/"cache",Detector(),row,"mug",frames)
            cache=root/"cache.json"
            write_json(cache,dict(passed=True,config=config,config_sha256=cache_key(config),collection_sha256="source",episodes=[entry]))
            boxes,_=read_boxes(manifest,cache)
            norm={"state":{"mean":[0]*17,"scale":[1]*17},"action":{"mean":[0]*8,"scale":[1]*8}}
            a,b=ThreeViewWindows(manifest,"train",None,norm),ThreeViewWindows(manifest,"train",boxes,norm)
            self.assertEqual(len(a),4)
            for i in range(4):
                sa,sb=a[i],b[i]
                for j in (0,1,2,3,5,6): np.testing.assert_array_equal(sa[j],sb[j])
                self.assertEqual(sa[1].shape,(3,224,224))
                np.testing.assert_array_equal(sa[4],sb[4][:3])
                self.assertEqual(int(sb[-1].sum()),4-i)
                self.assertEqual(bool(sb[4][3].any()),i==1)
                self.assertAlmostEqual(sb[5][0,0],.3+i*.01,places=6)
            np.testing.assert_array_equal(b[0][2][0], b[0][2][1])
            self.assertAlmostEqual(float(b[2][2][0,0]),.1,places=6)
            self.assertAlmostEqual(float(b[2][2][1,0]),.2,places=6)
            sa,sb=supervised_indices(a,32,147),supervised_indices(b,32,147)
            torch.testing.assert_close(sa,sb)
            audited=supervision(b,sb)["phases"]
            self.assertEqual(audited["close"]["sampled_unique"],2)
            self.assertEqual(audited["lift"]["loss_terms"],0)
            prediction=torch.zeros(1,8,8)
            labels=torch.from_numpy(b[3][5][None])
            mask=torch.from_numpy(b[3][-1][None])
            score=masked_chunk_loss(prediction,labels,mask).item()
            labels[:,1:]=100
            self.assertEqual(masked_chunk_loss(prediction,labels,mask).item(),score)

    def test_rotation_columns_are_invariant_to_quaternion_sign(self):
        a=proprioception(np.zeros(7),.03,[1,2,3,0,0,1,0])
        b=proprioception(np.zeros(7),.03,[1,2,3,0,0,-1,0])
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[-6:],[-1,0,0,0,-1,0])
