import inspect
import itertools
import multiprocessing
import re
from collections import OrderedDict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pandas_ta as pta
import talib as tta
from finta import TA as fta
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform
from tabulate import tabulate

from tuneta.config import *
from tuneta.optimize import Optimize
from tuneta.utils import col_name, distance_correlation

import os
import joblib

from tqdm.auto import tqdm 

# Distance correlation
def dc(p0, p1):
    df = pd.concat([p0, p1], axis=1).dropna()
    res = distance_correlation(
        np.array(df.iloc[:, 0]).astype(float), np.array(df.iloc[:, 1]).astype(float)
    )
    return res


class TuneTA:
    def __init__(self, n_jobs=1, verbose=False):
        self.fitted = []
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.remaining_tasks = {}
        self.completed_tasks = []

    def save_checkpoint(self, filename="checkpoint.pkl"):
        """Saves the current state of the fitting process."""
        checkpoint_data = {
            'fitted': self.fitted,
            'remaining_tasks': self.remaining_tasks,
            'completed_tasks': self.completed_tasks,
        }
        joblib.dump(checkpoint_data, filename)

    def load_checkpoint(self, filename="checkpoint.pkl"):
        """Loads the state of the fitting process from a checkpoint."""
        if os.path.exists(filename):
            checkpoint_data = joblib.load(filename)
            self.fitted = checkpoint_data.get('fitted', [])
            self.remaining_tasks = checkpoint_data.get('remaining_tasks', {})
            self.completed_tasks = checkpoint_data.get('completed_tasks', [])
            if self.verbose:
                print(f"Loaded checkpoint from {filename}")
        else:
            if self.verbose:
                print(f"No checkpoint found at {filename}")

    def fit(
        self,
        X,
        y,
        trials=5,
        indicators=["tta"],
        ranges=[(3, 180)],
        early_stop=99999,
        max_clusters=10,
        min_target_correlation=0.001,
        remove_consecutive_duplicates=False,
        checkpoint_interval=10,
        resume=False,
        checkpoint_file="checkpoint.pkl"
    ):
        """
        Optimize indicator parameters to maximize correlation
        :param X: Historical dataset
        :param y: Target used to measure correlation.  Can be a subset index of X
        :param trials: Number of optimization trials per indicator set
        :param indicators: List of indicators to optimize
        :param ranges: Parameter search space
        :param early_stop: Max number of optimization trials before stopping
        :param checkpoint_interval: Number of optimizations after which to save a checkpoint
        :param resume: Whether to resume from a checkpoint
        :param checkpoint_file: The file to save/load checkpoints
        """
        # Optionally load from a checkpoint
        if resume:
            self.load_checkpoint(checkpoint_file)

        # No missing values allowed
        if X.isna().any().any() or y.isna().any():
            raise ValueError("X and y cannot contain missing values")

        if not isinstance(X.index.get_level_values(0)[0], datetime):
            raise ValueError("Index must be of type datetime")

        if len(X) != len(y):
            raise ValueError("Length of X and y must be identical")

        self.fitted = []  # List containing each indicator completed study
        X.columns = X.columns.str.lower()  # columns must be lower case

        # Package level optimization
        if "tta" in indicators:
            indicators = indicators + talib_indicators
            indicators.remove("tta")
        if "pta" in indicators:
            indicators = indicators + pandas_ta_indicators
            indicators.remove("pta")
        if "fta" in indicators:
            indicators = indicators + finta_indicatrs
            indicators.remove("fta")
        if "all" in indicators:
            indicators = talib_indicators + pandas_ta_indicators + finta_indicatrs
        indicators = list(OrderedDict.fromkeys(indicators))

        # Create textual representation of function in Optuna format
        # Example: 'tta.RSI(X.close, length=trial.suggest_int(\'timeperiod1\', 2, 1500))'
        # Utilizes the signature of the indicator (ie user parameters) if available
        # TTA uses help docstrings as signature is not available in C bindings
        # Parameters contained in config.py are tuned

        # Iterate user defined search space ranges
        # tasks = []
        for low, high in ranges:
            if low <= 1:
                raise ValueError("Range low must be > 1")
            if high >= len(X):
                raise ValueError(f"Range high:{high} must be > length of X:{len(X)}")

            # Iterate indicators per range
            for ind in indicators:

                # Index column to optimize if indicator returns dataframe
                idx = 0
                if ":" in ind:
                    idx = int(ind.split(":")[1])
                    ind = ind.split(":")[0]
                fn = f"{ind}("

                # If TTA indicator, use doc strings for lack of better way to
                # get indicator arguments (C binding)
                if ind[0:3] == "tta":
                    usage = eval(f"{ind}.__doc__").split(")")[0].split("(")[1]
                    params = re.sub("[^0-9a-zA-Z_\s]", "", usage).split()

                # Pandas-TA and FinTA both can be inspected for parameters
                else:
                    sig = inspect.signature(eval(ind))
                    params = sig.parameters.values()

                # Format function string
                suggest = False
                for param in params:
                    param = re.split(":|=", str(param))[0].strip()
                    if param == "open_":
                        param = "open"
                    if param == "real":
                        fn += f"X.close, "
                    elif param == "ohlc" or param == "ohlcv":
                        fn += f"X, "
                    elif param in tune_series:
                        fn += f"X.{param}, "
                    elif param in tune_params:
                        suggest = True
                        if param == "mamode":
                            if "pta" in fn and not any(
                                [
                                    indicator
                                    for indicator in [
                                        "inertia",
                                        "qqe",
                                        "kama",
                                        "smma",
                                        "zlma",
                                        "rvi",
                                    ]
                                    if (indicator in fn)
                                ]
                            ):
                                fn += f"{param}=trial.suggest_categorical('{param}', {pandas_ta_mamodes}), "
                        else:
                            fn += (
                                f"{param}=trial.suggest_int('{param}', {low}, {high}), "
                            )
                if "pta" in fn:
                    fn += "lookahead=False, "
                fn += ")"

                # Only optimize indicators that contain tunable parameters
                if suggest:
                    self.remaining_tasks[fn] = delayed(Optimize(
                        function=fn,
                        n_trials=trials,
                        remove_consecutive_duplicates=remove_consecutive_duplicates,
                    ).fit)(X, y, idx=idx, max_clusters=max_clusters, verbose=self.verbose, early_stop=early_stop)
                else:
                    self.remaining_tasks[fn] = delayed(Optimize(
                        function=fn,
                        n_trials=1,
                        remove_consecutive_duplicates=remove_consecutive_duplicates,
                    ).fit)(X, y, idx=idx, max_clusters=max_clusters, verbose=self.verbose, early_stop=early_stop)

        # Progress bar
        completed_tasks = 0
        for task in self.completed_tasks:
            del self.remaining_tasks[task]
            completed_tasks += 1

        print(f"Found {completed_tasks} completed tasks from previous run.")
        
        pbar = tqdm(total=len(self.remaining_tasks), desc="Optimizing", unit="tasks")

        results = Parallel(n_jobs=self.n_jobs)(self.remaining_tasks.values())

        self.fitted = []
        for key, result in zip(self.remaining_tasks.keys(), results):
            completed_tasks += 1
            pbar.update(1)  # Update the progress bar
            if len(result.study.user_attrs) > 0:
                self.fitted.append(result)
                self.completed_tasks.append(key)  # Add the key of the completed task

            if completed_tasks % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_file)
                if self.verbose:
                    print(f"Checkpoint saved: {completed_tasks} tasks completed.")

        pbar.close()  # Close the progress bar

        if len(self.fitted) == 0:
            raise RuntimeError("No successful trials")

        # Remove any fits with less than minimum target correlation
        self.fitted = [
            f
            for f in self.fitted
            if f.study.user_attrs["best_trial"].user_attrs["correlation"]
            > min_target_correlation
        ]

        # Order fits by correlation (Descending)
        self.fitted = sorted(
            [f for f in self.fitted],
            key=lambda x: x.study.user_attrs["best_trial"].value,
            reverse=True,
        )


        return self

    def prune(self, max_inter_correlation=0.7, top_prior=99999, top_post=99999):
        """
        Select most correlated with target, least intercorrelated
        :param top: Selects top x most correlated with target
        :param studies: From top x, keep y least intercorelated
        :return:
        """

        fit_count = len(self.fitted)

        # Create feature correlation dataframe and remove duplicates
        feature_correlation = [
            [
                f.study.user_attrs["name"],
                f.study.user_attrs["best_trial"].user_attrs["correlation"],
            ]
            for f in self.fitted
        ]
        feature_correlation = pd.DataFrame(feature_correlation).sort_values(
            by=1, ascending=False
        )
        feature_correlation = feature_correlation.drop_duplicates(
            subset=0, keep="first"
        )  # Duplicate indicators
        feature_correlation[1] = feature_correlation[1].round(4)
        feature_correlation = feature_correlation.drop_duplicates(
            subset=[1], keep="first"
        )  # Duplicate correlation

        # Filter top correlated features
        feature_correlation = feature_correlation.head(top_prior)
        self.fitted = [
            f for i, f in enumerate(self.fitted) if i in feature_correlation.index
        ]

        if not hasattr(self, "f_corr") or fit_count != len(self.fitted):
            self.features_corr()

        # Iteratively removes least fit individual of most correlated pairs of studies
        # IOW, finds most correlated pairs, removes lest correlated to target until x studies
        components = list(range(len(self.fitted)))
        indices = list(range(len(self.fitted)))
        correlations = np.array(self.f_corr)

        most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
        correlation = correlations[most_correlated[0], most_correlated[1]]
        while correlation > max_inter_correlation:
            most_correlated = np.unravel_index(
                np.argmax(correlations), correlations.shape
            )
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))
            most_correlated = np.unravel_index(
                np.argmax(correlations), correlations.shape
            )
            correlation = correlations[most_correlated[0], most_correlated[1]]

        # Remove most correlated fits
        self.fitted = [self.fitted[i] for i in components][:top_post]

        # Recalculate correlation of fits
        self.target_corr()
        self.features_corr()
        return self


    def transform(self, X, columns=None):
        """
        Given X, create features of fitted studies
        :param X: Dataset with features used to create fitted studies
        :return: a DataFrame with the results
        """
        X.columns = X.columns.str.lower()  # columns must be lower case
        self.result = []

        # Create a Parallel object with the number of jobs
        parallel = Parallel(n_jobs=self.n_jobs)

        # Call the transform method for each fitted study in parallel using joblib's delayed function
        # Each call to delayed returns a function that will be called by joblib
        tasks = (delayed(ind.transform)(X) for ind in self.fitted)

        # Run the tasks in parallel and collect the results
        self.result = parallel(tasks)

        # Combine results into dataframe to return
        res = pd.concat(self.result, axis=1)
        return res

    def target_corr(self):
        fns = []  # Function names
        cor = []  # Target Correlation
        for fit in self.fitted:
            fns.append(
                col_name(fit.function, fit.study.user_attrs["best_trial"].params)
            )
            cor.append(np.round(fit.study.user_attrs["best_trial"].value, 6))

        # Target correlation
        self.t_corr = pd.DataFrame(cor, index=fns, columns=["Correlation"]).sort_values(
            by=["Correlation"], ascending=False
        )

    def features_corr(self):
        fns = []  # Function names
        cor = []  # Target Correlation
        features = []
        for fit in self.fitted:
            fns.append(
                col_name(fit.function, fit.study.user_attrs["best_trial"].params)
            )
            cor.append(np.round(fit.study.user_attrs["best_trial"].value, 6))
            features.append(fit.study.user_attrs["best_trial"].user_attrs["res_y"])

        # Feature must be same size for correlation and of type float
        start = max([f.first_valid_index() for f in features])
        features = [(f[f.index >= start]).astype(float) for f in features]

        # Inter Correlation
        pair_order_list = itertools.combinations(features, 2)
        correlations = Parallel(n_jobs=self.n_jobs)(
            delayed(dc)(p[0], p[1]) for p in pair_order_list
        )
        correlations = squareform(correlations)
        self.f_corr = pd.DataFrame(correlations, columns=fns, index=fns)
        return self

    def report(self, target_corr=True, features_corr=True):
        if target_corr:
            if not hasattr(self, "t_corr"):
                self.target_corr()
            print("\nIndicator Correlation to Target:\n")
            print(tabulate(self.t_corr, headers=self.t_corr.columns, tablefmt="simple"))

        if features_corr:
            if not hasattr(self, "f_corr"):
                self.features_corr()
            print("\nIndicator Correlation to Each Other:\n")
            print(tabulate(self.f_corr, headers=self.f_corr.columns, tablefmt="simple"))

    def fit_times(self):
        times = [fit.time for fit in self.fitted]
        inds = [fit.function.split("(")[0] for fit in self.fitted]
        df = pd.DataFrame({"Indicator": inds, "Times": times}).sort_values(
            by="Times", ascending=False
        )
        print(tabulate(df, headers=df.columns, tablefmt="simple"))

    def prune_df(
        self,
        X,
        y,
        min_target_correlation=0.001,
        max_inter_correlation=0.7,
        report=True,
        top_prior=99999,
        top_post=99999,
    ):
        if X.isna().any().any() or y.isna().any():
            raise ValueError("X and y cannot contain missing values")

        # Correlations to target
        tc = [distance_correlation(np.array(x[1]), np.array(y)) for x in X.iteritems()]
        names = [x[0] for x in X.iteritems()]
        target_correlation = pd.DataFrame(
            tc, index=names, columns=["Correlation"]
        ).sort_values(by=["Correlation"], ascending=False)

        # Columns greater than 0 correlation
        target_correlation = target_correlation[
            target_correlation.Correlation > min_target_correlation
        ]

        # Intercorrelation pruning is expensive.
        target_correlation = target_correlation.head(top_prior)

        if report:
            print("\nIndicator Correlation to Target:\n")
            print(
                tabulate(
                    target_correlation,
                    headers=target_correlation.columns,
                    tablefmt="simple",
                )
            )

        # Calculate inter correlation
        columns = target_correlation.index.values
        features = [x[1] for x in X[columns].iteritems()]
        pair_order_list = itertools.combinations(features, 2)
        correlations = Parallel(n_jobs=self.n_jobs)(
            delayed(dc)(p[0], p[1]) for p in pair_order_list
        )  # Parallelize correlation calculation
        correlations = squareform(correlations)
        components = list(range(len(correlations)))
        indices = list(range(len(correlations)))
        most_correlated = np.unravel_index(np.argmax(correlations), correlations.shape)
        correlation = correlations[most_correlated[0], most_correlated[1]]
        while correlation > max_inter_correlation:
            most_correlated = np.unravel_index(
                np.argmax(correlations), correlations.shape
            )
            worst = max(most_correlated)
            components.pop(worst)
            indices.remove(worst)
            correlations = correlations[:, indices][indices, :]
            indices = list(range(len(components)))
            most_correlated = np.unravel_index(
                np.argmax(correlations), correlations.shape
            )
            correlation = correlations[most_correlated[0], most_correlated[1]]

        # Get columns of features to keep
        if len(columns):
            columns = columns[components]

            # Report intercorrelation
            if report:
                correlations = pd.DataFrame(
                    correlations, columns=columns, index=columns
                )
                print("\nIntercorrelation after prune:\n")
                print(
                    tabulate(
                        correlations, headers=correlations.columns, tablefmt="simple"
                    )
                )

            # Correlations to target
            X = X[columns]
            tc = [
                distance_correlation(np.array(x[1]), np.array(y)) for x in X.iteritems()
            ]
            names = [x[0] for x in X.iteritems()]
            target_correlation = pd.DataFrame(
                tc, index=names, columns=["Correlation"]
            ).sort_values(by=["Correlation"], ascending=False)
            target_correlation = target_correlation.head(top_post)
            columns = target_correlation.index.values

        return columns
