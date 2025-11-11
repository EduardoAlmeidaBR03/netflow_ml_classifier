import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.tree import export_text
import os
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
        min_impurity_decrease=0.000003,
        min_samples_leaf=3,
        min_samples_split=9, 
        max_depth=25, 
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
    
    # Predição completa do conjunto de teste
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
    
    # Informações da estrutura da árvore
    n_nodes = classifier.tree_.node_count
    n_leaves = classifier.tree_.n_leaves
    max_depth = classifier.tree_.max_depth
    
    model_filename = "4-treinando_decision_tree/decision_tree_model.joblib"
    dump(classifier, model_filename)
    
    generate_confusion_matrix_svg(y_test, y_pred)
    
    return classification_report_result, confusion_matrix_result, n_nodes, n_leaves, max_depth


def generate_confusion_matrix_svg(y_test, y_pred):
    """Gera a matriz de confusão em formato SVG com valores em porcentagem"""
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, 
                cbar_kws={'label': 'Porcentagem (%)'}, 
                annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                square=False)
    
    plt.xlabel('Classe Predita', fontsize=16, fontweight='bold')
    plt.ylabel('Classe Real', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('4-treinando_decision_tree/confusion_matrix.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print("Matriz de confusão salva: confusion_matrix.svg")


def export_tree_structure(classifier, feature_names):
    """Exporta a estrutura da árvore para um arquivo de texto"""
    tree_rules = export_text(classifier, feature_names=feature_names, max_depth=10)
    
    output_file = "4-treinando_decision_tree/decision_tree_structure.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ESTRUTURA DA ÁRVORE DE DECISÃO\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuração do Modelo:\n")
        f.write(f"  - Critério: {classifier.criterion}\n")
        f.write(f"  - Max Depth: {classifier.max_depth}\n")
        f.write(f"  - Min Samples Split: {classifier.min_samples_split}\n")
        f.write(f"  - Min Samples Leaf: {classifier.min_samples_leaf}\n")
        f.write(f"  - Min Impurity Decrease: {classifier.min_impurity_decrease}\n\n")
        
        f.write(f"Estatísticas da Árvore:\n")
        f.write(f"  - Número Total de Nós: {classifier.tree_.node_count:,}\n")
        f.write(f"  - Número de Folhas: {classifier.tree_.n_leaves:,}\n")
        f.write(f"  - Profundidade Máxima Real: {classifier.tree_.max_depth}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("REGRAS DA ÁRVORE (Primeiros 10 níveis)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(tree_rules)
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("IMPORTÂNCIA DAS FEATURES\n")
        f.write("=" * 80 + "\n\n")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importância': classifier.feature_importances_
        }).sort_values('Importância', ascending=False)
        
        for idx, row in feature_importance.iterrows():
            f.write(f"{row['Feature']:<30} : {row['Importância']*100:.2f}%\n")
    
    print(f"Estrutura da árvore salva: {output_file}")
    return output_file

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
    
    class_report, cm, n_nodes, n_leaves, max_depth = evaluate_model(classifier, X_test, y_test, y_pred)
    

    
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
    
    print("\n" + "=" * 50)
    print("ESTRUTURA DA ÁRVORE")
    print("=" * 50)
    print(f"Número de Nós: {n_nodes:,}")
    print(f"Número de Folhas: {n_leaves:,}")
    print(f"Profundidade Máxima Real: {max_depth}")
    print(f"Profundidade Configurada: {classifier.max_depth}")
    
    print(f"\nModelo salvo: decision_tree_model.joblib")
    
    return metrics_dict['accuracy']

if __name__ == "__main__":
    accuracy = main()
