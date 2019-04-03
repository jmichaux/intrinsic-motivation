#https://gist.github.com/machinaut/209c44e8c55245c0d0f0094693053158
import os
import mujoco_py
import numpy as np

PATH_TO_HUMANOID_XML = os.path.expanduser('~/.mujoco/mjpro150/model/humanoid.xml')

# Load the model and make a simulator
model = mujoco_py.load_model_from_path(PATH_TO_HUMANOID_XML)
sim = mujoco_py.MjSim(model)

# Simulate 1000 steps so humanoid has fallen on the ground
for _ in range(10000):
    sim.step()

print('number of contacts', sim.data.ncon)
for i in range(sim.data.ncon):
    # Note that the contact array has more than `ncon` entries,
    # so be careful to only read the valid entries.
    contact = sim.data.contact[i]
    print('contact', i)
    # print('dist', contact.dist)
    print('geom1', contact.geom1, sim.model.geom_id2name(contact.geom1))
    print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
    # There's more stuff in the data structure
    # See the mujoco documentation for more info!
    geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
    print(' Contact force on geom2 body', sim.data.cfrc_ext[geom2_body])
    print('norm', np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))))
    # Use internal functions to read out mj_contactForce
    c_array = np.zeros(6, dtype=np.float64)
    print('c_array', c_array)
    mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)
    print('c_array', c_array)

print('done')
