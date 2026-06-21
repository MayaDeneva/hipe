from hipe.eval.metrics import macro_recall


def test_macro_recall_known_value():
    # 3 classes present in gold; predicting all FALSE.
    y_true = ["FALSE", "TRUE", "FALSE", "PROBABLE"]
    y_pred = ["FALSE", "FALSE", "FALSE", "FALSE"]
    # recall: FALSE=2/2=1, PROBABLE=0/1=0, TRUE=0/1=0 -> macro = 1/3
    assert round(macro_recall(y_true, y_pred), 4) == 0.3333


def test_macro_recall_perfect():
    y = ["TRUE", "FALSE", "FALSE"]
    assert macro_recall(y, y) == 1.0
