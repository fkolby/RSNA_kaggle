#https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369267  
def pfbeta_torch(preds, labels, beta=1):
    if preds.dim() != 2 or (preds.dim() == 2 and preds.shape[1] !=2): raise ValueError('Houston, we got a problem')
    preds = preds[:, 1]
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0

# https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369886    
def pfbeta_torch_thresh(preds, labels):
    optimized_preds = optimize_preds(preds, labels)
    return pfbeta_torch(optimized_preds, labels)

def optimize_preds(preds, labels=None, thresh=None, return_thresh=False, print_results=True):
    preds = preds.clone()
    if labels is not None: without_thresh = pfbeta_torch(preds, labels)
    
    if not thresh and labels is not None:
        threshs = np.linspace(0, 1, 101)
        f1s = [pfbeta_torch((preds > thr).float(), labels) for thr in threshs]
        idx = np.argmax(f1s)
        thresh, best_pfbeta = threshs[idx], f1s[idx]

    preds = (preds > thresh).float()

    if print_results:
        print(f'without optimization: {without_thresh}')
        pfbeta = pfbeta_torch(preds, labels)
        print(f'with optimization: {pfbeta}')
        print(f'best_thresh = {thresh}')
    if return_thresh:
        return thresh
    return preds

fn2label = {fn: cancer_or_not for fn, cancer_or_not in zip(train_csv['image_id'].astype('str'), train_csv['cancer'])}

def splitting_func(paths):
    train = []
    valid = []
    for idx, path in enumerate(paths):
        if int(path.parent.name) in patient_id_any_cancer.iloc[splits[SPLIT][0]].patient_id.values:
            train.append(idx)
        else:
            valid.append(idx)
    return train, valid

def label_func(path):
    return fn2label[path.stem]

def get_items(image_dir_path):
    
    items = []
    for p in get_image_files(image_dir_path):
        items.append(p)
        if p.stem in fn2label and int(p.parent.name) in patient_id_any_cancer.iloc[splits[SPLIT][0]].patient_id.values:
    
            if label_func(p) == 1:
                #this seems odd; if cancer, append 5 pictures?
                for _ in range(5):
                    items.append(p)
    return items