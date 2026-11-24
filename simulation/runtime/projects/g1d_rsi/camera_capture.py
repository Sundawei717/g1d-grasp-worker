"""Read native Isaac camera RGB and ReferenceTime without advancing physics.

Camera.get_rgba reads the RGB annotator directly. get_current_frame timestamps
can instead be frequency gated. Use the same native ReferenceTime mapping as
Camera._data_acquisition_callback (core nodes in 5.1, SimulationManager in 6.0),
and bind a copied image between
two matching reference stamps. Runtime validation is still required; CPU tests
of this module do not certify a working renderer.
"""
from __future__ import annotations

import importlib.metadata
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


SCHEMA = "g1d.native_camera_capture.v1"


class CameraCaptureError(RuntimeError):
    pass


def _diagnostic_plain(value):
    if isinstance(value, dict):
        return {str(k): _diagnostic_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostic_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _diagnostic_plain(value.tolist())
    if isinstance(value, np.generic):
        return _diagnostic_plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_native_value": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return {"numerator": int(value.numerator), "denominator": int(value.denominator)}
    return repr(value)


def _diagnostic_read(callback):
    try:
        return _diagnostic_plain(callback())
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def _diagnostic_runtime_state():
    """Inspect real settings and Fabric state; never author a value or pump Kit."""
    import carb.settings
    import omni.usd
    settings = carb.settings.get_settings()
    keys = ("/rtx/hydra/supportMultiTickRate", "/app/player/playSimulations", "/app/useFabricSceneDelegate",
            "/app/settings/fabricDefaultStageFrameHistoryCount", "/isaaclab/render/offscreen",
            "/isaaclab/render/rtx_sensors", "/isaaclab/cameras_enabled", "/physics/fabricUpdateTransformations",
            "/physics/updateToUsd", "/app/renderer/waitIdle", "/app/hydraEngine/waitIdle",
            "/omni/replicator/asyncRendering", "/app/updateOrder/checkForHydraRenderComplete",
            "/exts/omni.replicator.core/Orchestrator/enabled", "/omni/replicator/disableAnnotatorGate",
            "/omni/replicator/captureOnPlay", "/persistent/omni/replicator/captureOnPlay",
            "/app/asyncRendering")
    context = omni.usd.get_context()
    result = {"settings": {key: _diagnostic_read(lambda key=key: settings.get(key)) for key in keys},
              "usd_stage_id": _diagnostic_read(context.get_stage_id),
              "attached_hydra_engines": _diagnostic_read(context.get_attached_hydra_engine_names)}
    def read_external_time():
        import usdrt
        stage = usdrt.Usd.Stage.Attach(context.get_stage_id())
        prim = stage.GetPrimAtPath("/ExternalSimulationTime")
        if not prim.IsValid():
            return {"prim_valid": False, "source": "Fabric usdrt stage", "value_not_created": True}
        attribute = prim.GetAttribute("omni:time")
        return {"prim_valid": True, "attribute_valid": bool(attribute.IsValid()),
                "value": _diagnostic_plain(attribute.Get()) if attribute.IsValid() else None,
                "source": "Fabric /ExternalSimulationTime.omni:time", "value_not_created": True}
    result["external_simulation_time_fabric"] = _diagnostic_read(read_external_time)
    def read_lab_state():
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance()
        return {"is_rendering": bool(sim.is_rendering), "render_generation": sim.render_generation,
                "visualizers": [type(value).__module__ + "." + type(value).__qualname__ for value in sim.visualizers]}
    result["lab_context"] = _diagnostic_read(read_lab_state)
    def read_sdg_gate():
        import omni.graph.core as og
        nodes = {}
        # These are the actual installed Replicator DispatchSync/OgnRefTimeGate
        # ports. Reading them does not schedule captures or evaluate the graph.
        attributes = ("inputs:enabled", "inputs:simTimesToWrite", "inputs:prevSimTimesToWrite",
            "inputs:referenceTimeNumerator", "inputs:referenceTimeDenominator",
            "outputs:referenceTimeNumerator", "outputs:referenceTimeDenominator")
        for name in ("DispatchSync", "PostProcessDispatcher"):
            path = "/Render/PostProcess/SDGPipeline/" + name
            node = og.get_node_by_path(path)
            record = {"path": path, "present": bool(node and node.is_valid())}
            if record["present"]:
                record["compute_count"] = _diagnostic_read(node.get_compute_count)
                record["attributes"] = {}
                for attr_name in attributes:
                    attribute = node.get_attribute(attr_name)
                    record["attributes"][attr_name] = (_diagnostic_read(attribute.get)
                        if attribute and attribute.is_valid() else {"attribute_present": False})
            nodes[name] = record
        return nodes
    result["sdg_gate"] = _diagnostic_read(read_sdg_gate)
    return result


def _diagnostic_annotator(camera):
    """Expose a missing or malformed raw RGB buffer without converting it to an image."""
    annotators = getattr(camera, "_custom_annotators", {})
    rgb = annotators.get("rgb")
    result = {"attached_annotator_keys": list(annotators), "rgb_entry_present": rgb is not None,
              "camera_initialized": getattr(camera, "_is_initialized", None),
              "camera_acquisition_callback_present": getattr(camera, "_acquisition_callback", None) is not None,
              "camera_paused": getattr(camera, "_pause", None),
              "camera_previous_time": getattr(camera, "_previous_time", None),
              "camera_elapsed_time": getattr(camera, "_elapsed_time", None)}
    if rgb is None:
        return result
    result["annotator_type"] = type(rgb).__module__ + "." + type(rgb).__qualname__
    # Preserve relevant actual attachment fields only; do not invent field names
    # or dump the annotator's graph/runtime objects into the JSON report.
    result["attachment_fields"] = {name: _diagnostic_plain(value) for name, value in vars(rgb).items()
        if ("render_product" in name or "attached" in name) and isinstance(value, (str, bool, list, tuple, dict, type(None)))}
    data = rgb.get_data(device="cpu")
    def describe(value):
        if isinstance(value, dict):
            return {"python_type": "dict", "keys": list(value),
                    "values": {str(key): describe(item) for key, item in value.items()}}
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            return {"python_type": type(value).__module__ + "." + type(value).__qualname__,
                    "shape": list(value.shape), "ndim": int(value.ndim), "size": int(value.size),
                    "dtype": str(value.dtype), "pixel_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        return {"python_type": type(value).__module__ + "." + type(value).__qualname__,
                "value": _diagnostic_plain(value)}
    result["raw_get_data"] = describe(data)
    return result


def _diagnostic_reference_classification(reference, samples):
    try:
        query = Fraction(int(reference["referenceTimeNumerator"]), int(reference["referenceTimeDenominator"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return {"classification": "invalid_reference"}
    valid = [(i, Fraction(row["numerator"], row["denominator"])) for i, row in enumerate(samples)
             if row.get("valid") and row.get("denominator", 0) > 0]
    if not valid:
        return {"classification": "no_valid_samples", "query_seconds": float(query)}
    low = min(valid, key=lambda item: item[1])
    high = max(valid, key=lambda item: item[1])
    matches = [index for index, key in valid if key == query]
    category = ("exact_sample" if matches else "before_min" if query < low[1]
                else "after_max" if query > high[1] else "between_samples")
    return {"classification": category, "query_seconds": float(query), "matching_ring_indices": matches,
            "minimum_sample": samples[low[0]], "maximum_sample": samples[high[0]],
            "query_minus_minimum_reference_s": float(query - low[1]),
            "query_minus_maximum_reference_s": float(query - high[1])}


def diagnose_owned_cameras(cameras: dict, output, *, clock: Callable[[], object], reason: str = ""):
    """Export unaccepted native pixels/clock evidence after a failed capture.

    Performs reads only: no render, physics step, setting changes or timestamp
    repair. Use a fresh directory outside the accepted episode data. The caller
    must re-raise the original capture error even if this diagnostic succeeds.
    """
    from datetime import datetime, timezone
    from PIL import Image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "g1d.native_camera_failure_diagnostic.v1", "diagnostic_only": True,
              "accepted": False, "training_data_accepted": False, "render_called": False,
              "physics_step_called": False, "runtime_settings_changed": False, "reason": str(reason),
              "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "clock_before": _diagnostic_read(clock), "runtime_before": _diagnostic_read(_diagnostic_runtime_state),
              "cameras": {}}
    def persist():
        (output / "diagnostic.json").write_text(json.dumps(_diagnostic_plain(report), indent=2,
                                                          allow_nan=False) + "\n", encoding="utf-8")
    persist()
    for name, camera in cameras.items():
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Camera diagnostic filename must be a simple identifier")
        entry = {"accepted": False, "clock_before": _diagnostic_read(clock)}
        report["cameras"][name] = entry
        entry["render_product_path"] = _diagnostic_read(camera.get_render_product_path)
        entry["raw_rgb_annotator"] = _diagnostic_read(lambda: _diagnostic_annotator(camera))
        entry["raw_reference_before"] = _diagnostic_read(lambda: camera._fabric_time_annotator.get_data())
        try:
            mapper, source, provider, interface = _native_time_mapper(camera)
            entry.update(timestamp_source=source, clock_mapper=provider)
            if interface is not None:
                for method in ("get_simulation_time", "get_num_physics_steps", "get_current_time", "get_sample_count",
                               "get_sample_range", "get_buffer_capacity"):
                    entry[method] = _diagnostic_read(lambda method=method: getattr(interface, method)())
                capacity = entry["get_buffer_capacity"]
                if not isinstance(capacity, int) or not 0 < capacity <= 64:
                    raise CameraCaptureError("Refusing an unexpectedly large diagnostic sample ring")
                samples = [{"ring_index": i, "numerator": int(item.time.numerator),
                    "denominator": int(item.time.denominator), "sim_time_s": float(item.data.sim_time),
                    "sim_time_monotonic_s": float(item.data.sim_time_monotonic),
                    "system_time_s": float(item.data.system_time), "valid": bool(item.valid)}
                    for i, item in enumerate(interface.get_all_samples())]
                entry["full_native_samples"] = _diagnostic_plain(samples)
                entry["reference_vs_sample_range"] = _diagnostic_reference_classification(entry["raw_reference_before"], samples)
            raw = entry["raw_reference_before"]
            entry["native_mapping_return_unaccepted"] = _diagnostic_read(lambda: mapper((
                int(raw["referenceTimeNumerator"]), int(raw["referenceTimeDenominator"]))))
        except Exception as exc:
            entry["sample_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        copied_rgba = None
        try:
            pixels = camera.get_rgba(device="cpu")
            if hasattr(pixels, "numpy"):
                pixels = pixels.numpy()
            if pixels is None:
                entry["rgba"] = {"present": False}
            else:
                pixels = np.array(pixels, copy=True)
                copied_rgba = pixels
                target = output / f"{name}_rgba.npy"
                np.save(target, pixels, allow_pickle=False)
                entry["rgba"] = {"present": True, "shape": list(pixels.shape), "dtype": str(pixels.dtype),
                    "npy": target.name, "npy_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "decoded_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
                if pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] in (3, 4):
                    png = output / f"{name}_rgba.png"
                    Image.fromarray(pixels).save(png)
                    rgb = pixels[:, :, :3]
                    entry["rgba"].update(png=png.name, png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),
                        rgb_mean=float(rgb.mean()), rgb_spatial_std=float(rgb.astype(np.float32).std(axis=(0, 1)).mean()),
                        black_fraction=float((rgb.max(axis=2) <= 2).mean()),
                        white_fraction=float((rgb.min(axis=2) >= 253).mean()))
        except Exception as exc:
            entry["rgba_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        try:
            sdk_frame = camera.get_current_frame(clone=True)
            sdk_arrays = {}
            def sdk_value(value, key):
                if isinstance(value, dict):
                    return {str(k): sdk_value(v, key + "/" + str(k)) for k, v in value.items()}
                if isinstance(value, np.ndarray) or (hasattr(value, "numpy") and hasattr(value, "shape")):
                    array = np.array(value.numpy() if hasattr(value, "numpy") else value, copy=True)
                    summary = {"shape": list(array.shape), "dtype": str(array.dtype),
                               "pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                               "matches_get_rgba_exact": bool(copied_rgba is not None and np.array_equal(array, copied_rgba))}
                    if (copied_rgba is not None and array.ndim == copied_rgba.ndim == 3
                            and array.shape[:2] == copied_rgba.shape[:2] and array.shape[2] >= 3):
                        summary["matches_get_rgba_rgb"] = bool(np.array_equal(array[:, :, :3], copied_rgba[:, :, :3]))
                    sdk_arrays[key] = summary
                    return {"array_evidence_key": key}
                return _diagnostic_plain(value)
            entry["sdk_get_current_frame"] = {"clone": True,
                "actual_keys": list(sdk_frame) if isinstance(sdk_frame, dict) else None,
                "metadata": sdk_value(sdk_frame, "frame"), "arrays": sdk_arrays,
                "accepted_as_synchronized_pair": False}
        except Exception as exc:
            entry["sdk_current_frame_error"] = f"{type(exc).__name__}: {exc}"
        entry["raw_reference_after"] = _diagnostic_read(lambda: camera._fabric_time_annotator.get_data())
        entry["reference_unchanged_during_copy"] = entry["raw_reference_before"] == entry["raw_reference_after"]
        entry["clock_after"] = _diagnostic_read(clock)
        persist()
    report["clock_after"] = _diagnostic_read(clock)
    report["clock_unchanged"] = report["clock_before"] == report["clock_after"]
    report["runtime_after"] = _diagnostic_read(_diagnostic_runtime_state)
    report["diagnostic_export_complete"] = True
    persist()
    return report


def _native_time_mapper(camera):
    """Return the camera's actual mapping interface, including its cached manager.

    Lab 3 may replace the public SimulationManager alias after Camera imported
    its manager. Read that callback's globals instead of reimporting the alias;
    this is exactly the object used by the installed Camera callback itself.
    """
    nodes = getattr(camera, "_core_nodes_interface", None)
    if nodes is not None:
        method = getattr(nodes, "get_sim_time_at_time", None)
        if not callable(method):
            raise CameraCaptureError("Native core-nodes simulation-time mapper is unavailable")
        return method, "native_ReferenceTime_core_nodes_sim_time_mapping", {
            "interface": "Camera._core_nodes_interface", "method": "get_sim_time_at_time"}, None
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CameraCaptureError("Cannot identify a supported native camera clock provider") from exc
    if version != "6.0.1.0":
        raise CameraCaptureError(f"Native core-nodes clock absent in unsupported Isaac version {version}")
    callback = getattr(camera, "_data_acquisition_callback", None)
    callback_module = getattr(callback, "__module__", None)
    callback_globals = getattr(callback, "__globals__", {})
    if callback_module != "isaacsim.sensors.camera.camera":
        raise CameraCaptureError("Isaac6 camera callback is not the audited installed provider")
    manager = callback_globals.get("SimulationManager")
    interface = getattr(manager, "_simulation_manager_interface", None)
    method = getattr(interface, "get_simulation_time_at_time", None)
    if not callable(method):
        raise CameraCaptureError("Isaac6 camera SimulationManager native time mapper is unavailable")
    return method, "native_ReferenceTime_simulation_manager_sim_time_mapping", {
        "interface": "Camera._data_acquisition_callback.__globals__.SimulationManager._simulation_manager_interface",
        "method": "get_simulation_time_at_time", "camera_callback_module": callback_module,
        "manager_module": getattr(manager, "__module__", None),
        "manager_class": getattr(manager, "__qualname__", None), "wheel_version": version}, interface


def _multitick_enabled():
    """Read the same real setting consulted by native TimeSampleStorage."""
    import carb.settings
    return bool(carb.settings.get_settings().get_as_bool("/rtx/hydra/supportMultiTickRate"))


def _mapping_sample_evidence(interface, stamp, mapped_time):
    """Prove the native mapper did not silently return current time on a miss.

    Native 6.0 getSimulationTimeAtTime returns current physics time when storage
    is empty or a non-multitick reference is outside its range. A matching number
    alone is insufficient. Use read-only native sample APIs and the same exact /
    adjacent interpolation / multitick rules as TimeSampleStorage. No sample or
    runtime setting is authored here, and this never replaces the native result.
    """
    try:
        count_before = int(interface.get_sample_count())
        capacity = int(interface.get_buffer_capacity())
        # This installed 6.0.1 TimeSampleStorage has a compile-time capacity of
        # 31. Refuse a different implementation before materializing its ring.
        if capacity != 31 or not 0 < count_before <= capacity:
            raise CameraCaptureError("Native time-sample storage is empty or unexpected; current-time fallback is forbidden")
        entries = interface.get_all_samples()
        count_after = int(interface.get_sample_count())
        multitick = _multitick_enabled()
    except CameraCaptureError:
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise CameraCaptureError("Native time-sample evidence is unavailable") from exc
    if not isinstance(entries, (list, tuple)) or count_before != count_after or len(entries) != count_after:
        raise CameraCaptureError("Native time-sample storage changed while reading evidence")
    if not 0 < count_after <= capacity:
        raise CameraCaptureError("Native time-sample storage is empty; current-time fallback is forbidden")
    samples = []
    try:
        for entry in entries:
            n, d = int(entry.time.numerator), int(entry.time.denominator)
            value = float(entry.data.sim_time)
            if not entry.valid or d <= 0 or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid native time sample")
            samples.append({"numerator": n, "denominator": d, "sim_time_s": value})
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise CameraCaptureError("Malformed native time-sample evidence") from exc
    query = Fraction(*stamp)
    ring_sha = hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode("utf-8")).hexdigest()
    proof = {"schema": "g1d.native_time_sample_evidence.v1", "verified": True,
             "native_current_time_fallback_rejected": True,
             "multi_tick_setting": "/rtx/hydra/supportMultiTickRate", "multi_tick_enabled": multitick,
             "sample_count": count_after, "buffer_capacity": capacity,
             "sample_ring_sha256": ring_sha,
             "sample_ring_canonicalization": "json_sorted_keys_utf8_compact_v1",
             "sample_export_api": "ISimulationManager.get_all_samples",
             "maximum_samples_embedded": 2}
    keys = [Fraction(row["numerator"], row["denominator"]) for row in samples]
    if multitick:
        # Native getSimulationTimeAt directly returns double(rtime) on this
        # branch. The plugin-level getter still requires nonempty storage first.
        expected = float(query)
        chosen = sorted({min(range(len(keys)), key=lambda i: keys[i]), max(range(len(keys)), key=lambda i: keys[i])})
        proof.update(lookup_mode="multitick_rational_time", matched_sample_indices=[])
    else:
        matches = [i for i, key in enumerate(keys) if key == query]
        if matches:
            index = matches[-1]  # Native exact lookup searches newest to oldest.
            chosen = [index]
            expected = samples[index]["sim_time_s"]
            proof.update(lookup_mode="exact_sample", matched_sample_indices=[0])
        else:
            ordered = sorted(range(len(keys)), key=lambda i: keys[i])
            pair = next(((before, after) for before, after in zip(ordered, ordered[1:])
                         if keys[before] < query < keys[after]), None)
            if pair is None:
                raise CameraCaptureError(
                    "Native ReferenceTime is outside stored samples; current-time fallback is forbidden")
            before, after = pair
            # Match native C++ double arithmetic when reproducing its audit value.
            t0, t1 = float(keys[before]), float(keys[after])
            if t1 == t0:
                raise CameraCaptureError("Native interpolation interval lost floating-point precision")
            alpha = (float(query) - t0) / (t1 - t0)
            start, stop = samples[before]["sim_time_s"], samples[after]["sim_time_s"]
            expected = start + (stop - start) * alpha
            chosen = list(pair)
            proof.update(lookup_mode="adjacent_sample_interpolation", matched_sample_indices=[0, 1],
                         interpolation_alpha=alpha)
    if not math.isfinite(expected) or abs(expected - mapped_time) > 1e-8:
        raise CameraCaptureError("Native time mapping disagrees with its actual sample evidence")
    proof["sample_derived_sim_time_s"] = expected
    proof["native_mapped_sim_time_s"] = mapped_time
    proof["samples"] = [samples[i] for i in chosen]
    proof["sample_ring_indices"] = chosen
    return proof


def native_reference(camera):
    """Read the audited 5.1/6.0 Camera API's per-product simulation timestamp."""
    annotator = getattr(camera, "_fabric_time_annotator", None)
    if annotator is None:
        raise CameraCaptureError("Camera native ReferenceTime is not initialized")
    mapper, timestamp_source, provider, sample_interface = _native_time_mapper(camera)
    data = annotator.get_data()
    if not isinstance(data, dict):
        raise CameraCaptureError("Expected native ReferenceTime dictionary")
    try:
        numerator = int(data["referenceTimeNumerator"])
        denominator = int(data["referenceTimeDenominator"])
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise CameraCaptureError("Invalid native ReferenceTime fields") from exc
    if numerator < 0 or denominator <= 0:
        raise CameraCaptureError("Native ReferenceTime must be nonnegative with a positive denominator")
    try:
        sensor_time = float(mapper((numerator, denominator)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise CameraCaptureError("Native reference has no mapped sensor simulation time") from exc
    if not math.isfinite(sensor_time) or sensor_time < 0:
        raise CameraCaptureError("Invalid native sensor simulation time")
    product = camera.get_render_product_path()
    if not isinstance(product, str) or not product:
        raise CameraCaptureError("Native camera render product is not initialized")
    evidence = (_mapping_sample_evidence(sample_interface, (numerator, denominator), sensor_time)
                if sample_interface is not None else None)
    result = {"reference_time": {"numerator": numerator, "denominator": denominator},
            "sensor_sim_time_s": sensor_time,
            "render_product_path": product,
            "timestamp_source": timestamp_source,
            "clock_mapper": provider}
    if evidence is not None:
        result["mapping_sample_evidence"] = evidence
    return result


def capture_owned_cameras(cameras: dict, *, expected_sim_time: float,
                          render: Callable[[], None], sim_time: Callable[[], float],
                          previous: dict | None = None, max_passes: int = 8,
                          tolerance_s: float = 1e-5, resolutions: dict | None = None):
    """Fence, copy and verify all camera products at one physical state.

    A retry only calls render(), never take_action()/step(). A stale, missing or
    racing product cannot become a training frame. previous is the frames map
    from the preceding physical control step; images may legitimately be equal
    during holds, but native frame identities and physical time must advance.
    """
    expected_sim_time = float(expected_sim_time)
    if (not cameras or not math.isfinite(expected_sim_time) or expected_sim_time < 0
            or type(max_passes) is not int or max_passes < 1
            or not math.isfinite(tolerance_s) or tolerance_s <= 0):
        raise ValueError("Require cameras, finite physical time, positive retry/time limits")
    previous = previous or {}
    errors = []
    for attempt in range(1, max_passes + 1):
        before = float(sim_time())
        if not math.isfinite(before) or abs(before - expected_sim_time) > tolerance_s:
            raise CameraCaptureError("Physical time changed before camera capture")
        render()
        after = float(sim_time())
        if not math.isfinite(after) or abs(after - before) > 1e-9:
            raise CameraCaptureError("Render-only camera fence advanced physical time")
        images, frames = {}, {}
        try:
            for name, camera in cameras.items():
                first = native_reference(camera)
                rgba = camera.get_rgba(device="cpu")
                if hasattr(rgba, "numpy"):
                    rgba = rgba.numpy()
                rgba = np.array(rgba, copy=True)
                second = native_reference(camera)
                if first != second:
                    raise CameraCaptureError(f"{name}: native frame changed during RGB copy")
                width, height = (resolutions or {}).get(name, (640, 480))
                if rgba.shape != (height, width, 4) or rgba.dtype != np.uint8:
                    raise CameraCaptureError(f"{name}: invalid native RGBA {rgba.shape}/{rgba.dtype}")
                if abs(first["sensor_sim_time_s"] - expected_sim_time) > tolerance_s:
                    raise CameraCaptureError(f"{name}: sensor time {first['sensor_sim_time_s']} differs from physics {expected_sim_time}")
                old = previous.get(name)
                if old is not None and (first["reference_time"] == old["reference_time"]
                        or first["sensor_sim_time_s"] <= old["sensor_sim_time_s"]):
                    raise CameraCaptureError(f"{name}: stale or nonadvancing native camera frame")
                images[name] = rgba
                frames[name] = {**first, "requested_sim_time_s": expected_sim_time,
                                "rgba_shape": list(rgba.shape), "dtype": "uint8"}
            end = float(sim_time())
            if not math.isfinite(end) or abs(end - before) > 1e-9:
                raise CameraCaptureError("Physical time changed while reading camera products")
            return {"schema": SCHEMA, "sim_time_s": expected_sim_time,
                    "physics_advanced": False, "render_passes": attempt,
                    "images": images, "frames": frames}
        except CameraCaptureError as exc:
            errors.append(str(exc))
    raise CameraCaptureError(f"No synchronized native camera capture after {max_passes} render fences: {errors[-4:]}")
