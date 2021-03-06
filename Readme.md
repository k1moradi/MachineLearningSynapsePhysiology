# A library to make deep learning or random forest models of synaptic physiology
I designed this library as part of my Ph.D. project. It is around a thousand lines of Python code and based on TensorFlow and Scikit-Learn. It allows quick testing of the latest machine learning technologies to make a predictive model of synapses, and test the predictive power of models.

## Dependencies
|module|tested version|
|---|---|
|Python|3.7.6|
|jupyter|1.0.0|
|tensorflow|2.3|
|tensorflow-addons|0.11.1|
|scikit-learn|0.22.1|
|eli5|0.10.1|
|psutil|5.6.7|
|matplotlib|3.1.0|
|pandas|1.0.1|
|scipy|1.4.1|
|numpy|1.18.5|
|plotly|4.9.0|
|plotly-orca|1.3.0|

On Linux "xvfb" package (Virtual Framebuffer 'fake' X server) may be needed as well for plotly to function.

## Users Manual
Check [Readme.ipynb](https://github.com/k1moradi/MachineLearningSynapsePhysiology/blob/main/Readme.ipynb) for a quick start guide.

The main function is assess_model.

|function parameter|default value|description|
|---|---|---|
|train_features_filename|None|data features: optional number of columns. If tau_r, tau_f, and U in the target matrix are NA, then inter-stimulus interval (ISI) should be set to zero.|
|train_targets_filename |None|data targets: five synaptic parameters g, tau_d, tau_r, tau_f, and U. NA values should be replaced with zero already.|
|predict_data_startswith|None|Any number of files containing prediction features. The columns of features should be identical to the training features. Each row should be one potential connection.|
|potential_connections_df|None|A DataFrame containing the list of potential connections. Each row is linked to prediction feature data files.|
|numerical_columns|['Slice_Thickness', 'ISI', 'Temperature', 'Vm', 'Erev_GABA_B', 'Erev_NMDA', 'Erev_GABA_A', 'Erev_AMPA', 'Cai', 'Cao', 'Cli', 'Clo', 'Csi', 'H2PO4o', 'HCO3i', 'HCO3o', 'HEPESi', 'Ki', 'Ko', 'Mgi', 'Mgo', 'Nai', 'Nao', 'Bri', 'gluconatei', 'QX314i', 'ATPi', 'EGTAi', 'EGTAo', 'GTPi', 'OHi', 'SO4i', 'SO4o', 'phosphocreatinei', 'methanesulfonatei', 'acetatei', 'methylsulfatei', 'NMDGi', 'Trisi', 'CeSO4i', 'pyruvateo', 'TEAi', 'Bao', 'HPO4o', 'Age']| List of feature columns that need normalization.|
|passage_num|1|the number of times NA values are replaced with the interpolations. Is used just for naming the folders for convenience.|
|complete_stp|True|interpolate the NA values by changing zero ISIs to the mode of the non-zero ISIs in the training data.|
|isi_column_name|'ISI'|the name of feature column that contains ISI values.|
|validation_data|None|Data that is sent to Keras during training to test the model on every epoch.|
|jack_knife|False|Tests the model with the jack-knife method. Each time you run the model one row is set aside and the model is trained using the remaining data. The results are stored in a folder. counter.txt file keeps track of the row number that is processed already.|
|random_forest|False| If set to True a random forest model is made instead of deep learning.|
|matrix_plot_is_needed|False| If True, a matrix plot of the five synaptic parameters is being plotted for each of the predictions.|
|feature_importance_iterations|0|If an integer more than zero is being used, a feature importance bar chart is being plotted.|
|source_directory|None|You can save trained models in a folder and ask the program to load the models in the source folder and retrained them.|
|destination_directory|None|you can choose a custom folder to save the results.|
|template_model|None|a model is being loaded and being used as a template.|
|template_weights|None|only weights of an earlier model are being loaded.|
|num_nodes|[8192, 512, 128, 512, 8192]|You can define the number of nodes in each layer of your deep learning model (a multilayer perceptron).|
|activation|'mish'|activation function of nodes. Supported functions include: 'tanh', 'softsign', 'softmax', 'PReLU', 'ReLU', 'LeakyReLU', 'ELU', 'selu', 'swish', 'mish', 'gelu', 'lisht', 'isrlu', and 'relu'.|
|drop_out|[0.5, 0.5, 0.05, 0.5, 0.5]|dropout rate after each layer of the neural net.|
|batch_normalization|True|Enables batch_normalization. Works better if large batch sizes are used.|
|noise|0.2|By default adds 20% Gaussian noise after the input layer and noise/4=5% after the output layer.|
|L1_weight|None|L1 regularization weight.|
|L2_weight|1e-3|L2 regularization weight.|
|min_weight|None|minimum allowed weight between nodes of the network.|
|max_weight|1.0|maximum allowed weight between nodes of the network.|
|batch_size|2621|Training batch size. Using large batch_sizes leads to a faster training speed. The prediction power of the models trained with large batch sizes was also better in the case of synapses.|
|num_epochs|9999|maximum number of training epochs after which training ends.|
|repetitions|4|number of times convergence of the model is being tested by checking if the model has reached the goal values also by checking the bio-plausibility of the predictions|
|optimizer|'ADAMW'|The name of the optimizer. Supported methods: 'ADAMW', 'SGDW', 'ADAM', and 'SGD'. The default method worked the best in our tests.|
|lookahead|True|A secondary optimizer that controls the training. Slightly improved the results in our tests.|
|learning_rate|0.015|Learning rate of the optimizer. Large values speed up the training.|
|LROPpatience|100|If training did not improve after this number of epochs reduce the learning rate by a factor.|
|LROPfactor|0.9|Learning rate reduction factor. The new learning_rate = previous learning_rate times LROPfactor.|
|loss|'SMAPE'|The loss function that computes the model error. Supported functions: 'MAPE' (mean absolute percentage error), 'MLAPE' (mean log1p absolute percentage error), 'MSLAPE' (mean squared log1p absolute percentage error), 'SMAAPE' (mean arctan absolute percentage error), 'ML1APE' (mean soft_l1 absolute percentage error), and 'MSLE' (mean squared logarithmic error).|
|loss_threshold|29|Loss threshold after which bio-plausibility is being tested.|
|loss_goal|29|Loss goal after which training is being stopped.|
|negative_penalty_coef|0.0|experimentally penalizes negative predictions if MSLE is being used as a loss function and the output layer activation is non-sigmoid.|
|patience|500|number of epochs training continues after finding the minimum loss value that is bellow threshold as well.|
|maape_threshold|27|accuracy threshold like the loss that is tested with MAAPE function.|
|maape_goal|27|accuracy goal like loss goal|
|ob_threshold|100|how many out of bound predictions are allowed.|
|skip|200|skip epochs while plotting the training curves.|
|never_stop|False|used while retraining models from a source folder. If no other model remains in the folder continue checking the folder.|
|save_models|False|save the trained model at the end.|
|verbose|0|if set to one, more training logs would be generated.|
