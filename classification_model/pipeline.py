from sklearn.pipeline import Pipeline
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import CountFrequencyEncoder
import classification_model.processing.features as pp
from sklearn.ensemble import RandomForestClassifier
from classification_model.config.core import config


pipe = Pipeline(
    [
        (
            "replace_categories",
            pp.ReplaceCatogories(
                variables=config.model.vars_to_replace_category,
                list_of_category_to_leave=config.model.categories_to_leave,
                replace_with=config.model.replace_with,
            ),
        ),
        (
            "cast_na_on_string",
            pp.CastNaOnString(
                variables=config.model.vars,
                string=config.model.string_to_na,
                na=config.model.na_type,
            ),
        ),
        (
            "cast_type",
            pp.CastType(variables=config.model.vars_to_float, dtype=float),
        ),
        (
            "title_extractor",
            pp.TitleExtractor(
                variables=config.model.var_to_extract_title,
                list_of_new_col_names=config.model.var_name_of_title,
            ),
        ),
        (
            "mean_median_imputer",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model.vars_na_to_mean,
            ),
        ),
        (
            "freq_imputer",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.model.vars_na_to_mfreq,
            ),
        ),
        (
            "mapper",
            pp.Mapper(
                variables=config.model.vars_to_map,
                mappings=config.model.dicts_to_map,
            ),
        ),
        (
            "frequency_encoder",
            CountFrequencyEncoder(
                encoding_method="frequency",
                variables=config.model.vars_to_freq_encode,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=config.model.n_estimators,
                random_state=config.app.random_state,
            ),
        ),
    ]
)
