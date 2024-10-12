from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def recursive_feature_elimination(X, y, n_features):
    """Применяет RFE для выбора наиболее важных признаков."""
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=n_features)
    fit = rfe.fit(X, y)
    return fit.support_

def apply_pca(X, n_components=2):
    """Применяет метод главных компонент для сокращения размерности."""
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    return principal_components, pca.explained_variance_ratio_
