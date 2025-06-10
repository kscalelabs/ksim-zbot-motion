<div align="center">
<h1>K-Sim Gym</h1>
<p>Train and deploy your own humanoid robot controller in 700 lines of Python</p>
<h3>
  <a href="https://youtu.be/c64FnSvj8kQ">Tutorial</a> ·
  <a href="https://kscale.dev/benchmarks">Leaderboard</a> ·
  <a href="https://docs.kscale.dev/docs/quick-start#/">Documentation</a>
  <br />
  <a href="https://github.com/kscalelabs/ksim/tree/master/examples">K-Sim Examples</a> ·
  <a href="https://github.com/kscalelabs/kbot-joystick">Joystick Example</a>
</h3>

https://github.com/user-attachments/assets/82e5e998-1d62-43e2-ae52-864af6e72629

</div>

## Installation Notes

I was getting some mystery CUDA version errors with Jax at one point. This was fixed by using the installation command below:

```bash
pip install -r requirements.txt 'jax[cuda12]' 'jaxlib==0.6.0'
```

## Notes

- Turn left / right animation duration is 0.8 seconds, to turn 0.7 radians, meaning the angular velocity of the robot should be 0.875 radians / second.
- Walk animation duration is 0.8 seconds, with each pair of steps moving forward by 0.70132 meters, meaning the walking speed of the robot should be 0.87665 meters / second.
