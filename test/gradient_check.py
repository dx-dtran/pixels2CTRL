import unittest
import numpy as np

##########################################
# Mock or import your functions here
##########################################

# Suppose these come from your main PPO script:
from pong_ppo import (
    ppo_actor_grad_piecemeal,
    compute_critic_gradients,
    sigmoid,
)


##########################################
# Helper: numerical gradient
##########################################

def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient of scalar function f at x.
    x can be a np.ndarray. f(x) should return a scalar.
    """
    grad = np.zeros_like(x)
    # Flatten so we can iterate easily
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        fx_plus = f(x)

        x[idx] = old_val - eps
        fx_minus = f(x)

        grad[idx] = (fx_plus - fx_minus) / (2.0 * eps)
        x[idx] = old_val  # reset
        it.iternext()
    return grad


##########################################
# Test class
##########################################

class TestGradients(unittest.TestCase):

    def test_ppo_actor_grad_piecemeal(self):
        """
        Compare analytical dL/dlogits from ppo_actor_grad_piecemeal
        with a numerical gradient approach.
        """
        np.random.seed(42)

        # Fake data
        N = 5
        logits = np.random.randn(N)  # shape (N,)
        p = sigmoid(logits)
        old_probs = 0.5 + 0.1 * np.random.randn(N)  # shape (N,)
        old_probs = np.clip(old_probs, 0.1, 0.9)
        advantages = np.random.randn(N)
        actions = np.random.choice([2, 3], size=N)
        eps = 0.2
        ent_coeff = 0.01

        # We'll define a loss function w.r.t. logits
        def ppo_loss_wrt_logits(logits_):
            """
            Return scalar PPO objective so we can do numeric diff wrt logits_.
            We'll define: L = mean( min(ratio * adv, clipped_ratio * adv ) )
            with sign inverted for gradient consistency, plus entropy term.
            """
            p_ = sigmoid(logits_)
            # replicate the same logic for pi_new, ratio, clipped, etc.
            a2_mask = actions == 2
            pi_new = np.where(a2_mask, p_, 1 - p_)
            ratio = pi_new / old_probs
            surr1 = ratio * advantages
            clipped_ratio = np.clip(ratio, 1.0 - eps, 1.0 + eps)
            surr2 = clipped_ratio * advantages
            # negative sign because we do gradient of "loss"
            clipped_obj = -np.minimum(surr1, surr2).mean()

            # Bernoulli entropy
            ent = -np.mean(p_ * np.log(p_ + 1e-10) + (1 - p_) * np.log(1 - p_ + 1e-10))
            return clipped_obj + ent_coeff * ent

        # Analytical gradient
        dL_dlogits_analytical = ppo_actor_grad_piecemeal(
            p, old_probs, advantages, actions, eps, ent_coeff
        )  # shape (N,)

        # Numerical gradient
        dL_dlogits_numerical = numerical_gradient(ppo_loss_wrt_logits, logits)

        # Compare
        np.testing.assert_allclose(
            dL_dlogits_analytical, dL_dlogits_numerical, rtol=1e-5, atol=1e-5
        )

    def test_critic_gradients(self):
        """
        Compare analytical dL/dW, dL/db from compute_critic_gradients
        with numerical gradients for a tiny network.
        """
        np.random.seed(42)
        # Tiny critic net
        D = 3
        H = 2
        critic = {
            "W1": np.random.randn(H, D) * 0.01,
            "b1": np.zeros((H, 1)),
            "W2": np.random.randn(1, H) * 0.01,
            "b2": np.zeros((1, 1)),
        }

        # Fake data
        N = 4
        mb_obs = np.random.randn(D, N)
        mb_returns = np.random.randn(N)

        # 1) Analytical
        grads_analytical = compute_critic_gradients(critic, mb_obs, mb_returns)

        # 2) Numerical. We'll do each parameter matrix individually:
        def critic_loss_fn(critic_dict):
            """
            Return MSE = mean((v - R)^2) from the critic for the current param values.
            """

            def value_forward(model, x):
                z1 = np.dot(model["W1"], x) + model["b1"]
                h = np.maximum(z1, 0)
                v = np.dot(model["W2"], h) + model["b2"]
                return v, h

            v, _ = value_forward(critic_dict, mb_obs)
            v = v.squeeze(axis=0)  # shape(N,)
            error = v - mb_returns
            return np.mean(error ** 2)

            # (Note we omit the factor of 2 or 1/2 because it doesn't matter
            # for comparing the gradient directions; just be consistent with
            # your analytical definition.)

        # We'll define a small wrapper for each param so we can do numeric diff
        # w.r.t W1, b1, W2, b2, store them, and restore after.

        # Numerical gradient for W1
        W1_original = critic["W1"].copy()

        def fW1(W1_):
            critic_copy = {
                "W1": W1_,
                "b1": critic["b1"],
                "W2": critic["W2"],
                "b2": critic["b2"],
            }
            return critic_loss_fn(critic_copy)

        num_grad_W1 = numerical_gradient(fW1, W1_original)
        # Compare with grads_analytical["W1"]
        np.testing.assert_allclose(
            grads_analytical["W1"], num_grad_W1, rtol=1e-5, atol=1e-5
        )

        # Numerical gradient for b1
        b1_original = critic["b1"].copy()

        def fb1(b1_):
            critic_copy = {
                "W1": critic["W1"],
                "b1": b1_,
                "W2": critic["W2"],
                "b2": critic["b2"],
            }
            return critic_loss_fn(critic_copy)

        num_grad_b1 = numerical_gradient(fb1, b1_original)
        np.testing.assert_allclose(
            grads_analytical["b1"], num_grad_b1, rtol=1e-5, atol=1e-5
        )

        # Numerical gradient for W2
        W2_original = critic["W2"].copy()

        def fW2(W2_):
            critic_copy = {
                "W1": critic["W1"],
                "b1": critic["b1"],
                "W2": W2_,
                "b2": critic["b2"],
            }
            return critic_loss_fn(critic_copy)

        num_grad_W2 = numerical_gradient(fW2, W2_original)
        np.testing.assert_allclose(
            grads_analytical["W2"], num_grad_W2, rtol=1e-5, atol=1e-5
        )

        # Numerical gradient for b2
        b2_original = critic["b2"].copy()

        def fb2(b2_):
            critic_copy = {
                "W1": critic["W1"],
                "b1": critic["b1"],
                "W2": critic["W2"],
                "b2": b2_,
            }
            return critic_loss_fn(critic_copy)

        num_grad_b2 = numerical_gradient(fb2, b2_original)
        np.testing.assert_allclose(
            grads_analytical["b2"], num_grad_b2, rtol=1e-5, atol=1e-5
        )


##########################################
# Run tests
##########################################
if __name__ == '__main__':
    unittest.main()
