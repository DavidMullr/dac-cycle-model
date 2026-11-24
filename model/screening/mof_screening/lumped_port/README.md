# Lumped cycle engine

The lumped engine treats the electrically heated bed as well mixed. It shares the material laws and weather inputs with the spatial model, including the humid Purolite A110 law. Direct steam and the steam jacket use the spatial engine. The runtime translates the selected electrical case into the configuration used here.

`engine.py` defines the state balances. `weather_driver.py` advances the phases
and accounts for finite-horizon results. The remaining modules handle pressure
control, pump-inlet cooling, water properties and solver initialization.

The lumped representation does not resolve bed temperature or loading gradients.
A short run does not establish cyclic steady state, and model outputs do not
establish measured performance or hardware suitability.
