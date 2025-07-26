from data_manager import TimeSeriesDataManager
import unittest
import numpy as np
import torch
# ---------------------------
#       Unit Tests
# ---------------------------
class TestRNNDataHandler(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset with 20 timesteps and 2 features.
        self.num_timesteps = 20
        self.num_features = 10
        # Example data: numbers from 0 to (20*2 - 1) reshaped into (20, 2)
        self.data = np.arange(self.num_timesteps * self.num_features).reshape(self.num_timesteps, self.num_features)
        self.lookback = 5
        self.device = torch.device('cpu')
        self.handler = TimeSeriesDataManager(self.data, self.lookback, self.device)

    def test_basic_split(self):
        """
        Test basic functionality when using target_var_idx and predictor_vars.
        """
        target_var_idx = 0
        predictor_vars = 1
        train_split = 0.8

        X_train, Y_train, X_val, Y_val = self.handler.create_rnn_dataset_split(
            target_var_idx=target_var_idx,
            predictor_vars=predictor_vars,
            train_split=train_split
        )

        total_windows = self.num_timesteps - self.lookback  # 20 - 5 = 15 windows
        expected_train_samples = int(train_split * total_windows)
        expected_val_samples = total_windows - expected_train_samples

        # Check that the shapes match expectations.
        self.assertEqual(X_train.shape[0], expected_train_samples)
        self.assertEqual(Y_train.shape[0], expected_train_samples)
        self.assertEqual(X_val.shape[0], expected_val_samples)
        self.assertEqual(Y_val.shape[0], expected_val_samples)

    def test_override_predictors(self):
        """
        Test that the override_predictors parameter correctly replaces the predictors.
        """
        target_var_idx = 0
        # predictor_vars won't be used because we override predictors.
        predictor_vars = 1  
        train_split = 0.5

        # Create an override predictors array with a different shape and constant value.
        override_predictors = np.full(self.num_timesteps, fill_value=42, dtype=np.float32)
        X_train, Y_train, X_val, Y_val = self.handler.create_rnn_dataset_split(
            target_var_idx=target_var_idx,
            predictor_vars=predictor_vars,
            train_split=train_split,
            override_predictors=override_predictors
        )

        # Check that all values in X_train and X_val are 42.
        self.assertTrue(torch.all(X_train == 42))
        self.assertTrue(torch.all(X_val == 42))

    def test_override_target(self):
        """
        Test that the override_target parameter correctly replaces the target values.
        """
        # In this test, override_target should be used in place of self.data[:, target_var_idx].
        target_var_idx = 0  # This will be ignored.
        predictor_vars = 1
        train_split = 1

        # Create an override target that is easy to verify.
        override_target = np.arange(self.num_timesteps, dtype=np.float32) * 10  # e.g., [0, 10, 20, ..., 190]
        X_train, Y_train, X_val, Y_val = self.handler.create_rnn_dataset_split(
            target_var_idx=target_var_idx,
            predictor_vars=predictor_vars,
            train_split=train_split,
            override_target=override_target
        )

        total_windows = self.num_timesteps - self.lookback
        # The targets used should be override_target[self.lookback: self.lookback + total_windows]
        expected_targets = override_target[self.lookback:self.lookback + total_windows]

        # Concatenate training and validation targets for comparison.
        Y_all = torch.cat([Y_train, Y_val]).cpu().numpy()
        np.testing.assert_array_equal(Y_all, expected_targets)

    def test_error_when_no_target(self):
        """
        Test that a ValueError is raised if neither target_var_idx nor override_target is provided.
        """
        with self.assertRaises(ValueError):
            self.handler.create_rnn_dataset_split(
                target_var_idx=None,
                predictor_vars=1,
                train_split=0.5,
                override_target=None
            )

    def test_error_when_no_predictors(self):
        """
        Test that a ValueError is raised if neither predictor_vars nor override_predictors is provided.
        """
        with self.assertRaises(ValueError):
            self.handler.create_rnn_dataset_split(
                target_var_idx=0,
                predictor_vars=None,
                train_split=0.5,
                override_predictors=None
            )
    def test_methods_equivalence(self):

        target_var_idx = 0
        # predictor_vars won't be used because we override predictors.
        predictor_vars = [4,2]  
        train_split = 0.5

        X_train, Y_train, X_val, Y_val = self.handler.create_rnn_dataset_split(
            target_var_idx=target_var_idx,
            predictor_vars=predictor_vars,
            train_split=train_split
        )

        X_train_old, Y_train_old = self.handler.create_rnn_dataset(
            target_var_idx=target_var_idx,
            predictor_vars=predictor_vars,
            start_idx=0,
            end_idx= int(self.num_timesteps*train_split)

        )

        X_val_old, Y_val_old = self.handler.create_rnn_dataset(
            target_var_idx=target_var_idx,
            predictor_vars=predictor_vars,
            start_idx=int(self.num_timesteps*train_split)

        )

        self.assertEqual(X_train.shape, X_train_old.shape, f"X_train shape {X_train.shape} is different from {X_train_old.shape}")
        self.assertEqual(Y_train.shape, Y_train_old.shape, f"Y_train shape {Y_train.shape} is different from {Y_train_old.shape}")
        self.assertEqual(X_val.shape, X_val_old.shape, f"X_val shape {X_val.shape} is different from {X_val_old.shape}")
        self.assertEqual(Y_val.shape, Y_val_old.shape, f"Y_train shape {Y_val.shape} is different from {Y_val_old.shape}")
        









if __name__ == '__main__':
    unittest.main()
