import lightgbm as lgb

def get_lightgbm_model(
    n_estimators: int = 150,
    learning_rate: float = 0.08,
    num_leaves: int = 31,
    max_depth: int = -1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    class_weight: str = None,
    random_state: int = 42
):
    """
    Retourne un classifieur LightGBM optimisé pour la classification multiclasse (27 classes).
    Gère nativement les matrices creuses TF-IDF et les variables denses.
    """
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=27,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
