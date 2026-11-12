"""Unit test: payload point mass conserves total mass / system COM / inertia about origin (myolegs22+Hippo_L1)."""
import numpy as np, mujoco
from myoassist_utils.env_spec import EnvSpec
from rl_train.envs import env_perturbations as ep
xml = EnvSpec(msk="myolegs22", device="Hippo_L1", terrain=None).validate().compose(planar_root=True)
m = mujoco.MjModel.from_xml_string(xml)
b = m.body("torso").id
def world_inertia_about_origin(m, b):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.body_iquat[b]); R = R.reshape(3,3)
    I = R @ np.diag(m.body_inertia[b]) @ R.T; c = m.body_ipos[b]; M = m.body_mass[b]
    return I + M*(c@c*np.eye(3) - np.outer(c,c)), M*c
I0, mc0 = world_inertia_about_origin(m, b); M0 = m.body_mass[b]
for mode in ("back", "front"):
    mm = mujoco.MjModel.from_xml_string(xml)
    ep.apply_payload(mm, 20.0, mode)
    I1, mc1 = world_inertia_about_origin(mm, b)
    r = ep.PAYLOAD_POS[mode]
    dI = 20.0*(r@r*np.eye(3)-np.outer(r,r))
    print(mode, "dmass", mm.body_mass[b]-M0, "dMC err", np.abs((mc1-mc0)-20*r).max(), "dI err", np.abs((I1-I0)-dI).max(),
          "inertia>0", (mm.body_inertia[b] > 0).all(), "total", mm.body_mass.sum())

# write-through check on the dm_control wrapper used inside MyoSuite envs
from dm_control import mujoco as dmm
ph = dmm.Physics.from_xml_string(xml)
before = float(ph.model.body_mass.sum())
ep.apply_payload(ph.model, 20.0, "back")
from rl_train.envs.env_perturbations import apply_population
from population import sample
apply_population(ph.model, sample(1, seed=3)[0], seed=3)
print("wrapper mass before", before, "after payload+population", float(ph.model.body_mass.sum()))
