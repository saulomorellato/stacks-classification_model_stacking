## PACOTES ##

library(tidyverse)	# manipulacao de dados
library(tidymodels) # ferramentas de ML
library(stacks)     # stacking
library(tictoc)     # registrar o tempo de execução de comandos



## DADOS ##

df<- read.csv("winequality_red.csv") %>% 
  data.frame()

glimpse(df)

df<- df %>% mutate(quality=ifelse(quality>=6,"good","bad"))
df$quality<- df$quality %>% as.factor()
glimpse(df)




##### SPLIT TRAIN/TEST/VALIDATION #####

set.seed(0)
split<- initial_split(df, strata=quality)

df.train<- training(split)
df.test<- testing(split)

folds<- vfold_cv(df.train, v=3, strata=quality)




##### PRÉ-PROCESSAMENTO #####

receita<- recipe(quality ~ . , data = df.train) %>%
  step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())




##### MODELOS #####

model_knn<- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

model_net<- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

model_rfo<- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")




##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_knn)

wf_net<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_net)

wf_rfo<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_rfo)



##### TUNAGEM DE HIPERPARAMETROS - BAYESIAN SEARCH #####

## KNN

tic()
tune_knn<- tune_bayes(wf_knn,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(neighbors(range=c(1,30)))
)
toc()
# 15.89 sec elapsed




## NET - ELASTIC NET

tic()
tune_net<- tune_bayes(wf_net,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(penalty(range=c(-10,10)),
                                              mixture(range=c(0,1)))
)
toc()
# 84.97 sec elapsed




## RFO - RANDOM FOREST

tic()
tune_rfo<- tune_bayes(wf_rfo,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(mtry(range=c(1,10)),
                                              trees(range=c(50,10000)),
                                              min_n(range=c(1,40)))
)
toc()
# 245.79 sec elapsed (~4min)





##### PREPARANDO STACKING #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_knn) %>% 
  add_candidates(tune_net) %>% 
  add_candidates(tune_rfo)

stack_ensemble_data





##### AJUSTANDO STACKING #####

set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = 10^(-9:-1),
                    mixture = 1, # 0=RIDGE; 1=LASSO
                    control = control_grid(),
                    non_negative = TRUE,
                    metric = metric_set(roc_auc))

autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights")

stack_ensemble_model




##### FINALIZANDO O MODELO #####

stack_ensemble_model<- stack_ensemble_model %>% 
  fit_members()

stack_ensemble_model




##### FINALIZANDO MODELOS INDIVIDUAIS #####

wf_train_knn<- wf_knn %>% finalize_workflow(select_best(tune_knn)) %>% fit(df.train)
wf_train_net<- wf_net %>% finalize_workflow(select_best(tune_net)) %>% fit(df.train)
wf_train_rfo<- wf_rfo %>% finalize_workflow(select_best(tune_rfo)) %>% fit(df.train)





#############################################
#####  CLASSIFICACAO SEM PONTO DE CORTE #####
#############################################

pred.knn<- wf_train_knn %>% predict(df.test, type="class")
pred.net<- wf_train_net %>% predict(df.test, type="class")
pred.rfo<- wf_train_rfo %>% predict(df.test, type="class")
pred.stc<- stack_ensemble_model %>% predict(df.test, type="class")

df.pred<- cbind.data.frame(df.test$quality,
                           pred.knn,
                           pred.net,
                           pred.rfo,
                           pred.stc)

colnames(df.pred)<- c("quality",
                      "knn",
                      "net",
                      "rfo",
                      "stc")


#####  VERIFICANDO MEDIDAS DE CLASSIFICAÇÃO  #####

medidas<- df.pred %>% 
  conf_mat(quality, knn) %>% 
  summary() %>% 
  select(-.estimator)

medidas<- cbind(summary(conf_mat(df.pred, quality, knn))[,-2],
                summary(conf_mat(df.pred, quality, net))[,3],
                summary(conf_mat(df.pred, quality, rfo))[,3],
                summary(conf_mat(df.pred, quality, stc))[,3])                     

colnames(medidas)<- c("medida",
                      "knn",
                      "net",
                      "rfo",
                      "stc")

medidas




