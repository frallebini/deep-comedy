import tensorflow as tf
import matplotlib.pyplot as plt


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Define a custom learning rate scheduler such that

        l_rate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    which corresponds to increasing the learning rate linearly for the first `warmup_steps` training steps, and
    decreasing it thereafter proportionally to the inverse square root of the step number.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    temp_learning_rate_schedule = CustomSchedule(128)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()
