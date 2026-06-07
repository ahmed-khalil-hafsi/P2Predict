from rich.console import Console

from p2predict.training import ALGORITHMS, _budget_params, _tune, build_pipeline, should_log_target

console = Console()


def hyper_parameter_tuning(
    X_train, y_train, numerical_cols, categorical_cols, algorithm,
    budget="fast", time_aware=False, log_target=None,
):
    """Tune the given algorithm and return the refitted best pipeline."""
    if log_target is None:
        log_target = should_log_target(y_train)
    pipeline = build_pipeline(
        algorithm, numerical_cols, categorical_cols, log_target=log_target
    )
    best_model, best_score = _tune(
        pipeline, X_train, y_train, algorithm, budget, log_target,
        time_aware=time_aware,
    )
    console.print(f"Tuned {algorithm} --> CV R²: {round(best_score, 3)}")
    return best_model, best_score, log_target


def compare_all_algorithms(
    X_train, y_train, numerical_cols, categorical_cols, budget="fast"
):
    """Tune every supported algorithm and report the best."""
    log_target = should_log_target(y_train)
    best_score = float("-inf")
    best = None
    for algorithm in ALGORITHMS:
        pipeline = build_pipeline(
            algorithm, numerical_cols, categorical_cols, log_target=log_target
        )
        model, score = _tune(
            pipeline, X_train, y_train, algorithm, budget, log_target
        )
        console.print(f"Model: {algorithm} --> CV R²: {round(score, 3)}")
        if score > best_score:
            best_score = score
            best = (algorithm, model)
    return best, best_score, log_target
