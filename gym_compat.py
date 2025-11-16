"""
Compatibility wrapper for gym 0.26+ to work with old gym 0.19 API
Allows code expecting old gym API (reset returns state, step returns 4-tuple) 
to work with gym 0.26+ (reset returns (state, info), step returns 5-tuple)

Usage: Import this at the top of your main script before importing gym
    import gym_compat  # Must be first
    import gym
    # Rest of code works as before
"""
import gym
from gym import spaces, Wrapper

# Store original gym.make
_original_make = gym.make


class CompatibilityWrapper(Wrapper):
    """Wrapper to convert gym 0.26+ API back to gym 0.19 API"""
    
    def reset(self, **kwargs):
        """Override reset to return only state (not tuple)"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result[0]  # Return only state, drop info
        return result
    
    def step(self, action):
        """Override step to return 4-tuple (not 5-tuple)"""
        result = self.env.step(action)
        if len(result) == 5:
            # New API: (state, reward, terminated, truncated, info)
            state, reward, terminated, truncated, info = result
            # Old API: (state, reward, done, info)
            done = terminated or truncated
            return state, reward, done, info
        return result


def _compat_make(*args, **kwargs):
    """Wrapper around gym.make to automatically apply compatibility wrapper"""
    env = _original_make(*args, **kwargs)
    return CompatibilityWrapper(env)


# Monkey patch gym.make
gym.make = _compat_make

print("✓ gym 0.26+ compatibility layer activated (old gym 0.19 API)")
