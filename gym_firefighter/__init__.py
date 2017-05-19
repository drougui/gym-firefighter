from gym.envs.registration import register

register(
    id='firefighter-v0',
    entry_point='gym_firefighter.envs:FirefighterEnv',
)
