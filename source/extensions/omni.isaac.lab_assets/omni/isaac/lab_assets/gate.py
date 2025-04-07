"""Configuration for the gates"""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils.assets import LOCAL_OBJECT_DIR

##
# Configuration
##

RECTANGLE_GATE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Gate",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_OBJECT_DIR}/Gate/rectangle_gate.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
        ),
        copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.0, 0.0, 0.5),
    ),
)
"""Configuration for the rectangle gate."""
