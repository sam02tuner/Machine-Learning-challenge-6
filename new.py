
import numpy as np
import pandas as pd


from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from lightgbm import LGBMClassifier


building_structure=pd.read_csv("F:/Dataset/Building_Structure.csv",index_col=None)
building_ownership=pd.read_csv("F:/Dataset/Building_Ownership_Use.csv",index_col=None)
train=pd.read_csv("F:/Dataset/train.csv",index_col=None)
test=pd.read_csv("F:/Dataset/test.csv",index_col=None)


result = pd.merge(building_structure, building_ownership, on='building_id')
res_train=pd.merge(train,result,on="building_id")
res_test=pd.merge(test,result,on="building_id")
index_train=res_train.pop('building_id')
train.index=index_train
index_test=res_test.pop('building_id')
test.index=index_test
cleanup_nums = {"area_assesed":{"Both": 1, "Building removed": 2,"Exterior":3, "Interior":4, "Not able to inspect":-999} ,
                "legal_ownership_status":{ "Institutional":1, "Other":-999, "Private":2, "Public":3 },
                "land_surface_condition":{ "Flat":1, "Moderate slope":2, "Steep slope": 3},
                "foundation_type": {"Bamboo/Timber":1, "Cement-Stone/Brick":2, "Mud mortar-Stone/Brick":3, "Other":-999, "RC":4},
                "roof_type": {"Bamboo/Timber-Heavy roof":1,"Bamboo/Timber-Light roof":2, "RCC/RB/RBC":3 },
                "ground_floor_type": { "Brick/Stone":1, "Mud":2,"Other":-999, "RC":3, "Timber":4},
                "other_floor_type": {"Not applicable":-999, "RCC/RB/RBC":1, "TImber/Bamboo-Mud":2, "Timber-Planck":3},
                "position": {"Attached-1 side":1,"Attached-2 side":2,"Attached-3 side":3, "Not attached":-999, "(Blanks)":-999},
                "plan_configuration": {"Building with Central Courtyard":1, "E-shape":2,"H-shape":3,"L-shape":4, "Multi-projected":5, "Others":-999, "Rectangular":7, "Square":8, "T-shape":9, "U-shape":10, "(Blanks)":-999},
                "condition_post_eq": {"Covered by landslide":1, "Damaged-Not used":2, "Damaged-Repaired and used":3, "Damaged-Rubble clear":4, "Damaged-Rubble Clear-New building built":5,  "Damaged-Rubble unclear":6, "Damaged-Used in risk":7, "Not damaged":-999 } }
res_train.replace(cleanup_nums, inplace=True)
res_test.replace(cleanup_nums, inplace=True)
clean_train={ "damage_grade": { "Grade 1":1,"Grade 2":2, "Grade 3":3, "Grade 4":4, "Grade 5":5 }}
res_train.replace(clean_train, inplace=True)
feature=res_train.pop('damage_grade')
res_train.fillna(-999,inplace=True)
res_test.fillna(-999,inplace=True)

scaler=StandardScaler()
res_train[['plinth_area_sq_ft']] = scaler.fit_transform(res_train[[ 'plinth_area_sq_ft']])
res_test[['plinth_area_sq_ft']] = scaler.fit_transform(res_test[[ 'plinth_area_sq_ft']])


"""
validation_size = 0.60
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(res_train, feature, test_size=validation_size, random_state=seed)

seed = 7
scoring = 'accuracy'

kfold = model_selection.KFold(n_splits=2, random_state=seed)
"""

model=LGBMClassifier(boosting='dart',num_leaves=452,num_iterations=5500,learning_rate=0.01,min_data_in_leaf=17, max_bin=800, bagging_fraction=0.74, max_depth=50,objective='binary')

"""
grid = GridSearchCV(model,param_grid)
grid.fit(res_train, feature)
# summarize the results of the grid search
print(grid.best_params_)
"""

model.fit(res_train, feature)

y_pred = model.predict(res_test)

my_submission = pd.DataFrame({'building_id': index_test, 'damage_grade': y_pred})
clean_submission={ "damage_grade": { 1:"Grade 1",2:"Grade 2",3:"Grade 3",4:"Grade 4",5:"Grade 5"}}
my_submission.replace(clean_submission, inplace=True)
#y_pred = np.round(y_pred,0)
my_submission.to_csv('F:/Dataset/submissionchallenge@@15.csv', index=False)
