 # Multi-modal Intrinsic Motivation for Robotic Exploration


 ## Requirements

 * Python 3
 * [PyTorch](http://pytorch.org/)
 * [OpenAI Gym](https://github.com/openai/gym)
 * [OpenAI baselines](https://github.com/openai/baselines)

 # Project Milestones
 - [ ] Modify gym environments
 	- [x] Modify gym environments to return images and depth map
 	- [ ] Modify orientation of additional cameras
 	- [ ] Return contact forces for the gripper
 - [ ] Implement PPO algorithm
	- [x] Implement Multimodal rollout container
 	- [ ] Implement PPO agent with FF policy
 	- [ ] Add Recurrent policy to PPO agent
 - [ ] Setup logging for the project
 - [ ] Setup the intrinsic curiosity Module
    - [ ] Forward Dynamics Model
    - [ ] Inverse Dynamics Model
    - [ ] VAE
    - [ ] Random CNN network
