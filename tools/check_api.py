import habitat_sim

print("Checking habitat_sim API...")
print("\nMain attributes with 'Sensor' or 'Camera':")
print([attr for attr in dir(habitat_sim) if 'Sensor' in attr or 'Camera' in attr])

print("\nChecking sensor module:")
if hasattr(habitat_sim, 'sensor'):
    print([attr for attr in dir(habitat_sim.sensor) if not attr.startswith('_')])

print("\nChecking bindings:")
if hasattr(habitat_sim, 'bindings'):
    print([attr for attr in dir(habitat_sim.bindings) if 'Sensor' in attr])
