UPPER_PART = "upper_side_panels"
LOWER_PART = "lower_side_panels"
MID_PART = "middle_side_panels"
TOP_PART = "top_side_panels"
MUG_BODY = "mug_body"
MUG_HANDLE = "mug_handle"
BASKET_HANDLE = "basket_handle"
BASKET_RIM = "basket_rim"
BOX_NEG_END = "box_negative_end"
BOX_POS_END = "box_positive_end"
PARTS = (UPPER_PART, LOWER_PART, MID_PART, TOP_PART, MUG_BODY, MUG_HANDLE,
         BASKET_HANDLE, BASKET_RIM, BOX_NEG_END, BOX_POS_END)


def contact_band(part, lower_z, upper_z):
    if part in {MUG_BODY, MUG_HANDLE, BASKET_HANDLE, BASKET_RIM, BOX_NEG_END, BOX_POS_END}:
        return lower_z, upper_z
    if part == UPPER_PART:
        return lower_z + 0.04, upper_z + 0.003
    if part == LOWER_PART and upper_z - lower_z >= 0.08:
        return lower_z + 0.022, lower_z + 0.038
    if part == MID_PART and upper_z - lower_z >= 0.10:
        return lower_z + 0.047, lower_z + 0.065
    if part == TOP_PART and upper_z - lower_z >= 0.10:
        return upper_z - 0.028, upper_z + 0.003
    raise ValueError(f"Unsupported grasp region for this asset: {part}")


def candidate_depths(part, lower_z, upper_z):
    if part == UPPER_PART:
        return 0.012, 0.018, 0.024, 0.008
    if part == TOP_PART:
        return 0.007, 0.009, 0.011
    low, high = contact_band(part, lower_z, upper_z)
    midpoint = (low + high) / 2
    if part == MID_PART:
        # The fingertips in the physical diagnostic contacted about 12 mm below the planner's grip-center target.
        return tuple(upper_z - midpoint - offset for offset in (0.012, 0.010, 0.014))
    return tuple(upper_z - anchor for anchor in (midpoint, high - 0.002, low + 0.002))


def anchor_height(part, lower_z, upper_z):
    if part == UPPER_PART:
        return upper_z - 0.018
    low, high = contact_band(part, lower_z, upper_z)
    return (low + high) / 2
