# Lumped cycle engine

This engine treats the bed as well mixed. It shares the material laws, weather,
equipment inputs and actuator policy used by the spatial model. The runtime
translates the selected case into its configuration.

`engine.py` defines the state balances. `weather_driver.py` advances the phases
and accounts for finite-horizon results. The remaining modules handle pressure
control, pump-inlet cooling, water properties and solver initialization.

The lumped representation does not resolve bed temperature or loading gradients.
A short run does not establish cyclic steady state, and model outputs do not
establish measured performance or hardware suitability.
