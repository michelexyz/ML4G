import numpy as np

class RegimeDataGenerator:
    """
    Generates synthetic time-series data with piecewise (regime-based) linear 
    causal structure. Each regime can have distinct min/max block lengths.
    We never pick the same regime twice in a row if choosing randomly.
    """

    def __init__(
        self, 
        regime_definitions: dict,
        n_vars: int,
        T: int,
        seed: int = 43,
        noise_std: float = 1.0,
        cycle_regimes: bool = True,
        add_fresh_noise: bool = False
    ):
        """
        Args:
            regime_definitions: A dict describing each regime. 
                For example:

                  regime_definitions = {
                      'Winter': {
                          'min_length': 30,
                          'max_length': 120,
                          'relationships': {
                              1: {
                                  0: {0: 0.4},
                                  1: {1: 0.3, 0: 0.7},
                                  2: {2: 0.3, 0: 0.7}
                              }
                          }
                      },
                      'Summer': {
                          'min_length': 20,
                          'max_length': 80,
                          'relationships': {
                              1: {
                                  0: {0: 0.4},
                                  1: {1: 0.3, 0: -0.7},
                                  2: {2: 0.3}
                              }
                          }
                      }
                      # etc.
                  }
                
                The 'relationships' entry for each regime has a structure:
                  { lag: { child_var: { parent_var: weight } } }
            
            n_vars: Number of variables (columns).
            T: Total time steps to generate.
            seed: random seed for reproducibility.
            noise_std: std dev of Gaussian noise added each step.
            cycle_regimes: if True, cycle them in a round-robin style,
                           if False, pick a random regime each time, 
                           excluding the immediate previous regime.
        """
        self.regime_definitions = regime_definitions
        self.n_vars = n_vars
        self.T = T
        self.seed = seed
        self.noise_std = noise_std
        self.cycle_regimes = cycle_regimes
        self.add_fresh_noise = add_fresh_noise

        self.regime_names = sorted(list(regime_definitions.keys()))
        
        
    def generate_data(self):
        """
        Generates the synthetic data with regime changes.
        Returns:
            data: shape (T, n_vars)
            regime_assignments: length T array with the regime name for each time index
        """
        np.random.seed(self.seed)
        data = np.random.randn(self.T, self.n_vars) * self.noise_std
        regime_assignments = []

        t = 0
        regime_idx = 0  # used only if cycle_regimes=True
        last_regime_name = None

        # store the number of switches between regimes
        number_of_switches = -1

        while t < self.T:

            number_of_switches += 1
            if self.cycle_regimes:
                # Pick regime in round-robin style
                regime_name = self.regime_names[regime_idx % len(self.regime_names)]
                regime_idx += 1
            else:
                # Pick random regime, excluding the last one
                possible_choices = [r for r in self.regime_names if r != last_regime_name]
                regime_name = np.random.choice(possible_choices)

            reg_def = self.regime_definitions[regime_name]
            min_len = reg_def['min_length']
            max_len = reg_def['max_length']
            relationships = reg_def['relationships']

            block_length = np.random.randint(min_len, max_len + 1)
            block_length = min(block_length, self.T - t)  # clamp so we don't exceed T

            # We'll fill data in the range [t, t+block_length)
            for step in range(t, t+block_length):
                # For each possible lag L in the regime
                for L, rel_dict_for_lag in relationships.items():
                    if step - L < 0:
                        # Not enough history
                        continue
                    # For each child variable v
                    for v, parent_dict in rel_dict_for_lag.items():
                        increment = 0.0
                        # sum the influences of each parent
                        for p, w in parent_dict.items():
                            increment += w * data[step - L, p]
                        data[step, v] += increment

                # add fresh noise
                if self.add_fresh_noise:
                    data[step, :] += np.random.randn(self.n_vars) * self.noise_std

            # Mark regime assignments
            regime_assignments.extend([regime_name] * block_length)


            t += block_length
            last_regime_name = regime_name
        
        # Convert assignments to array
        regime_assignments = np.array(regime_assignments)

        return data, regime_assignments, number_of_switches
    
    def create_mask_for_regime(self,
        regime_assignments: np.ndarray,
        regime_name: str
    ) -> np.ndarray:
        """
        Returns a boolean mask (T, n_vars) that is True exactly at the timesteps 
        where regime_assignments == regime_name, and False elsewhere.

        Args:
            regime_assignments: length T array of strings (e.g. ["Winter", "Winter", "Summer", ...]).
            regime_name: the name of the regime for which we want a mask (e.g. "Winter").
            n_vars: number of variables (to set the second dimension of the mask).

        Returns:
            mask: shape (T, n_vars), True for times in the given regime, False otherwise.
        """
        mask = np.zeros((self.T, self.n_vars), dtype=bool)
        mask[regime_assignments == regime_name, :] = True
        return mask


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    regime_definitions = {
        'Winter': {
            'min_length': 10,
            'max_length': 20,
            'relationships': {
                1: {
                    0: {0: 0.4},
                    1: {1: 0.3, 0: 0.7},
                    2: {2: 0.3, 0: 0.7}
                }
            }
        },
        'Summer': {
            'min_length': 10,
            'max_length': 20,
            'relationships': {
                1: {
                    0: {0: 0.4},
                    1: {1: 0.3, 0: -0.7},
                    2: {2: 0.3}
                }
            }
        }
    }

    generator = RegimeDataGenerator(
        regime_definitions=regime_definitions,
        n_vars=3,
        T=500,
        seed=43,
        noise_std=1.0,
        cycle_regimes=False,  # pick randomly, excluding last regime
    )

    data, assignments, number_of_switches= generator.generate_data()

    print("Data shape:", data.shape)
    print("Assignments shape:", assignments.shape)
    print("Unique regimes:", np.unique(assignments, return_counts=True))
    print("Number of switches between regimes:", number_of_switches)

    mask = generator.create_mask_for_regime(assignments, "Winter")
    print("Mask shape:", mask.shape)


