from pprint import pprint
import constants
from feast import FeatureStore

# Initialize the feature store
store = FeatureStore(repo_path=constants.REPO_PATH)
# Get serving data from feature store, we retrieve
# all features


feature_vector = store.get_online_features(
    features=[
        "diabete:pregnancies",
        "diabete:glucose",
        "diabete:bloodpressure",
        "diabete:skinthickness",  
        "diabete:insulin",
        "diabete:bmi",
        "diabete:diabetespedigreefunction",
        "diabete:age",      
    ],
    entity_rows=[
        {"diabete_id": 3},
    ],
).to_dict()

pprint(feature_vector)
