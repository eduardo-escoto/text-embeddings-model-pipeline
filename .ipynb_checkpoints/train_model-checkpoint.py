def build_seqs_pred(text, col, params, model_path):
    tokenizer = None
    with open(model_path + '/tokenizers/text_tokenizer_' + col + '.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    sequences = tokenizer.texts_to_sequences(text)
    sequences_padded = sequence.pad_sequences(sequences, maxlen = params['max_length'], padding='post', truncating='post')
    
    return sequences_padded  
    
def build_seqs(text_train, text_val, text_test, col, params, model_path):
    tokenizer = Tokenizer(num_words=params['max_words'], oov_token = '<OOV>')
    tokenizer.fit_on_texts(text_train)

    train_sequences = tokenizer.texts_to_sequences(text_train)
    train_sequences_padded = sequence.pad_sequences(train_sequences, maxlen=params['max_length'], padding='post', truncating='post')
    
    val_sequences = tokenizer.texts_to_sequences(text_val)
    val_sequences_padded = sequence.pad_sequences(val_sequences, maxlen=params['max_length'], padding='post', truncating='post')

    test_sequences = tokenizer.texts_to_sequences(text_test)
    test_sequences_padded = sequence.pad_sequences(test_sequences, maxlen=params['max_length'], padding='post', truncating='post')
    
    with open(model_path + '/tokenizers/text_tokenizer_' + col + '.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return train_sequences_padded, val_sequences_padded, test_sequences_padded    
    
def make_callbacks(params):
    early_stp = EarlyStopping(monitor='val_mse', min_delta=100, 
                            patience=5, verbose=1, mode='auto', restore_best_weights=True)
    reducer = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=2, min_delta=100, mode='min')
    decay = LearningRateScheduler(scheduler)
    if params['lr_modifier'] == 'decay':
        return [early_stp, decay]
    elif params['lr_modifier'] == 'reducer':
        return [early_stp, reducer]
    else:
        return [early_stp]
    
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr/epoch

def build_text_model(inp, params):
    text_model = (Embedding(params['max_words'], params['embedding_dim'], input_length = params['max_length']))(inp)
    
    for layer_num in np.arange(0, params['n_layers_gru']):
        layer_size = int(params['embedding_dim'] / (2**layer_num))
        text_model = (GRU(layer_size, 
                          return_sequences = not (layer_num == (params['n_layers_gru'] - 1))))(text_model)
        
    text_model = (Dense(8, activation = params['activation']))(text_model)
    
    return text_model
    
def build_nlp_model(train_shape, nlp_len, params):
    inp_feats = Input(shape = (train_shape,))
    inp_tknzd = lambda: Input(shape = (params['max_length'], ))

    model = (Dense(2**(params['n_layers'] + 2), activation = params['activation']))(inp_feats)
    model = (Dropout(params['dropout']))(model)
        
    for layer_num in np.flip(np.arange(2, params['n_layers'] + 1)):
        layer_size = 2**(layer_num + 1)
        model = (Dense(layer_size, activation = params['activation']))(model)
        if layer_size > 100:
            model = (Dropout(params['dropout']))(model)

    models = [model]  
    
    # Building as many text models as there are in nlp seqs    
    text_inputs = [inp_tknzd() for n in range(nlp_len)]
    text_models = [build_text_model(inp, params) for inp in text_inputs]
    
    model = concatenate(models + text_models)
    model = (Dense(1, activation=params['output_activation']))(model)
    model = Model(inputs = [inp_feats] + text_inputs, outputs = model)
    
    optimizer = None
    if params['optimizer'] =='SGD':
        optimizer=SGD(learning_rate=params['learning_rate'])
    elif params['optimizer'] =='Adam':
        optimizer=Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] =='RMSprop':
        optimizer=RMSprop(learning_rate=params['learning_rate'])
    elif params['optimizer'] =='Adagrad':
        optimizer=Adagrad(learning_rate=params['learning_rate'])
    elif params['optimizer'] =='Adamax':
        optimizer=Adamax(learning_rate=params['learning_rate'])
    elif params['optimizer'] =='Nadam':
        optimizer=Nadam(learning_rate=params['learning_rate'])

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    
    return model
    
def split_data_by_type(df, process_config):
    X = df[process_config['numeric_features'] + process_config['cat_ohe_features']].copy()
    y = df[process_config['target_feature']].copy()
    nlp = df[process_config['nlp_features']].copy()
    util = df[process_config['utility_features'] + [process_config['date_feature']]].copy()
    
    return X, y, nlp, util
 
def prep_data_pred(pred, model_path, process_config, model_config):
    process_config['cat_ohe_features'] = [
        x for x in pred.columns if any([y in x for y in process_config['categoric_features']])]
    
    X, y, nlp, util = split_data_by_type(pred, process_config)
        
    nlp_seqs = []
    for col in nlp.columns:
        seq = build_seqs_pred(nlp[col], col, model_config, model_path)
        nlp_seqs.append(seq)
        
    return (X, y, nlp_seqs, util)
    
    
def prep_data(train, val, test, model_path, process_config, model_config):
    process_config['cat_ohe_features'] = [
        x for x in train.columns if any([y in x for y in process_config['categoric_features']])]
    
    X_train, y_train, nlp_train, util_train = split_data_by_type(train, process_config)
    X_val, y_val, nlp_val, util_val = split_data_by_type(val, process_config)
    X_test, y_test, nlp_test, util_test =  split_data_by_type(test, process_config)
    
    nlp_seqs_train = []
    nlp_seqs_val = []
    nlp_seqs_test = []
    for col in nlp_train.columns:
        seq_train, seq_val, seq_test = build_seqs(nlp_train[col], nlp_val[col], nlp_test[col], col, model_config, model_path)
        nlp_seqs_train.append(seq_train)
        nlp_seqs_val.append(seq_val)
        nlp_seqs_test.append(seq_test)
        
    return (X_train, y_train, nlp_seqs_train, util_train,
            X_val, y_val, nlp_seqs_val, util_val,
            X_test, y_test, nlp_seqs_test, util_test)

def run_model(train, val, test, model_path, process_config, model_config):
    X_train, y_train, nlp_seqs_train, util_train, \
    X_val, y_val, nlp_seqs_val, util_val, \
    X_test, y_test, nlp_seqs_test, util_test = prep_data(train, val, test, model_path, process_config, model_config)
    
    model = build_nlp_model(X_train.shape[1], len(nlp_seqs_train), model_config)
    model.fit(
        [X_train] + nlp_seqs_train, 
        y_train, 
        validation_data = ([X_val] + nlp_seqs_val, y_val),
        epochs = model_config['epochs'], 
        batch_size = model_config['batch_size'],
        callbacks = make_callbacks(model_config), 
        verbose = model_config['verbose']
    )
    
    model.save(model_path + "/model")
    
    train_preds = model.predict([X_train] + nlp_seqs_train)
    val_preds = model.predict([X_val] + nlp_seqs_val)
    test_preds = model.predict([X_test] + nlp_seqs_test)
        
    util_train['target'] = y_train
    util_val['target'] = y_val
    util_test['target'] = y_test
    
    util_train['prediction'] = train_preds
    util_val['prediction'] = val_preds
    util_test['prediction'] = test_preds
    
    return model, util_train, util_val, util_test

def run_model_predict(pred, model_path, process_config, model_config):
    X, y, nlp_seqs, util = prep_data_pred(pred, model_path, process_config, model_config)

    model = tf.keras.models.load_model(model_path + "/model")
    
    preds = model.predict([X] + nlp_seqs)
        
    util['target'] = y
    util['prediction'] = preds
    
    return model, util