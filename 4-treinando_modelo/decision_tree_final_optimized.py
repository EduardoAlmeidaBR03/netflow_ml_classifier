import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    confusion_matrix, 
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score
)
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Carrega e preprocessa os dados"""
    chunk_size = 100000
    processed_chunks = []
    
    required_columns = [
        "Protocol", "Total.Fwd.Packets", "Total.Backward.Packets",
        "Total.Length.of.Bwd.Packets", "Total.Length.of.Fwd.Packets",
        "Fwd.Packet.Length.Mean", "Bwd.Packet.Length.Mean", "Average.Packet.Size",
        "Fwd.Packets.s", "Bwd.Packets.s", "Flow.Bytes.s", "Bwd.bytes.packet",
        "Fwd.bytes.packet", "Bwd.bytes.s", "Fwd.bytes.s", "Packets.s",
        "Speed.packet", "Fwd.Speed.packet", "Bwd.Speed.packet",
        "Ratio.Bytes", "Ratio.Packets", "ProtocolName"
    ]

    chunks = pd.read_csv("3-balanceado_dataset/dataset_tratado_balanceado.csv", chunksize=chunk_size)
    for chunk in chunks:
        chunk = chunk[required_columns]
        processed_chunks.append(chunk)
    
    final_dataframe = pd.concat(processed_chunks, ignore_index=True)
    final_dataframe = final_dataframe[final_dataframe["Protocol"] != 0]
    final_dataframe = final_dataframe[final_dataframe["ProtocolName"] != "OTHER"]
    
    return final_dataframe

def get_best_model():
    """Retorna o modelo com a melhor configuração encontrada"""
    best_classifier = DecisionTreeClassifier(
        criterion='entropy', 
        min_impurity_decrease=0.000001, 
        min_samples_leaf=3,
        min_samples_split=7, 
        max_depth=28, 
        max_features=None, 
        splitter='best', 
        random_state=42
    )
    
    return best_classifier

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Treina e avalia o modelo Decision Tree"""
    print("Treinando modelo Decision Tree...")
    
    classifier = get_best_model()
    
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = classifier.predict(X_test)
    
    metrics_dict = {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'training_time': training_time
    }
    
    try:
        y_pred_proba = classifier.predict_proba(X_test)
        metrics_dict['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, 
                                                   multi_class='ovr', average='macro')
    except:
        metrics_dict['roc_auc_ovr'] = None
    
    return classifier, y_pred, metrics_dict

def evaluate_model(classifier, X_test, y_test, y_pred):
    """Avalia o modelo e retorna métricas e importância das features"""
    classification_report_result = classification_report(y_test, y_pred, output_dict=True)
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    
    model_filename = "4-treinando_modelo/decision_tree_model.joblib"
    dump(classifier, model_filename)
    
    return classification_report_result, confusion_matrix_result

def main():
    """Função principal para treinar e avaliar o modelo Decision Tree"""
    print("Decision Tree - Treinamento e Avaliação")
    print("=" * 50)
    
    dataframe = load_and_preprocess_data()
    print(f"Dataset carregado: {len(dataframe):,} registros")
    
    feature_columns = [
        "Protocol", "Total.Fwd.Packets", "Total.Backward.Packets",
        "Total.Length.of.Bwd.Packets", "Total.Length.of.Fwd.Packets",
        "Fwd.Packet.Length.Mean", "Bwd.Packet.Length.Mean", "Average.Packet.Size",
        "Fwd.Packets.s", "Bwd.Packets.s", "Flow.Bytes.s", "Bwd.bytes.packet",
        "Fwd.bytes.packet", "Bwd.bytes.s", "Fwd.bytes.s", "Packets.s",
        "Speed.packet", "Fwd.Speed.packet", "Bwd.Speed.packet",
        "Ratio.Bytes", "Ratio.Packets"
    ]
    
    X = dataframe[feature_columns]
    y = dataframe["ProtocolName"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    classifier, y_pred, metrics_dict = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    class_report, cm = evaluate_model(classifier, X_test, y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("RESULTADOS")
    print("=" * 50)
    print(f"Acurácia Geral: {metrics_dict['accuracy']:.4f} ({metrics_dict['accuracy']*100:.2f}%)")
    print(f"F1-Score Macro: {metrics_dict['f1_macro']:.4f}")
    print(f"F1-Score Weighted: {metrics_dict['f1_weighted']:.4f}")
    print(f"Precisão Macro: {metrics_dict['precision_macro']:.4f}")
    print(f"Recall Macro: {metrics_dict['recall_macro']:.4f}")
    print(f"Tempo de treinamento: {metrics_dict['training_time']:.2f}s")
    
    if metrics_dict['roc_auc_ovr']:
        print(f"ROC AUC: {metrics_dict['roc_auc_ovr']:.4f}")
    
    print(f"\nModelo salvo: decision_tree_model.joblib")
    
    return metrics_dict['accuracy']

if __name__ == "__main__":
    accuracy = main()
