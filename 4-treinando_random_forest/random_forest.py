import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
    best_classifier = RandomForestClassifier(
        n_estimators=100,
        criterion='entropy',
        min_samples_split=2,
        min_samples_leaf=2,
        min_impurity_decrease=0.00001,
        max_features='sqrt',
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    return best_classifier

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Treina e avalia o modelo Random Forest"""
    print("Treinando modelo Random Forest...")
    
    classifier = get_best_model()
    
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    if hasattr(classifier, 'oob_score_') and classifier.oob_score_:
        print(f"📊 Out-of-bag Score: {classifier.oob_score_:.4f} ({classifier.oob_score_*100:.2f}%)")
    
    y_pred = classifier.predict(X_test)
    
    metrics_dict = {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'training_time': training_time,
        'oob_score': classifier.oob_score_ if hasattr(classifier, 'oob_score_') else None
    }
    
    try:
        y_pred_proba = classifier.predict_proba(X_test)
        metrics_dict['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, 
                                                    multi_class='ovr', average='macro')
    except:
        metrics_dict['roc_auc_ovr'] = None
    
    return classifier, y_pred, metrics_dict

def evaluate_model(classifier, X_test, y_test, y_pred):
    """Avalia o modelo e retorna métricas"""
    classification_report_result = classification_report(y_test, y_pred, output_dict=True)
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    
    # Informações da floresta
    n_estimators = classifier.n_estimators
    n_features = classifier.n_features_in_
    
    model_filename = "4-treinando_random_forest/random_forest_model.joblib"
    dump(classifier, model_filename)
    
    generate_confusion_matrix_svg(y_test, y_pred)
    
    return classification_report_result, confusion_matrix_result, n_estimators, n_features

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
    plt.savefig('4-treinando_random_forest/confusion_matrix.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print("Matriz de confusão salva: confusion_matrix.svg")

def export_forest_structure(classifier, feature_names):
    """Exporta a estrutura do Random Forest para um arquivo de texto"""
    output_file = "4-treinando_random_forest/random_forest_structure.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ESTRUTURA DO RANDOM FOREST\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuração do Modelo:\n")
        f.write(f"  - Critério: {classifier.criterion}\n")
        f.write(f"  - Número de Estimadores: {classifier.n_estimators}\n")
        f.write(f"  - Max Features: {classifier.max_features}\n")
        f.write(f"  - Min Samples Split: {classifier.min_samples_split}\n")
        f.write(f"  - Min Samples Leaf: {classifier.min_samples_leaf}\n")
        f.write(f"  - Min Impurity Decrease: {classifier.min_impurity_decrease}\n")
        f.write(f"  - Bootstrap: {classifier.bootstrap}\n")
        f.write(f"  - OOB Score: {classifier.oob_score}\n")
        f.write(f"  - Random State: {classifier.random_state}\n")
        f.write(f"  - N Jobs: {classifier.n_jobs}\n\n")
        
        # Calcular estatísticas das árvores
        total_nodes = sum(tree.tree_.node_count for tree in classifier.estimators_)
        total_leaves = sum(tree.tree_.n_leaves for tree in classifier.estimators_)
        max_depth_trees = [tree.tree_.max_depth for tree in classifier.estimators_]
        avg_depth = sum(max_depth_trees) / len(max_depth_trees)
        max_depth = max(max_depth_trees)
        min_depth = min(max_depth_trees)
        
        f.write(f"Estatísticas da Floresta:\n")
        f.write(f"  - Número de Árvores: {classifier.n_estimators}\n")
        f.write(f"  - Número de Features: {classifier.n_features_in_}\n")
        f.write(f"  - Número de Classes: {len(classifier.classes_)}\n")
        f.write(f"  - Total de Nós (todas as árvores): {total_nodes:,}\n")
        f.write(f"  - Total de Folhas (todas as árvores): {total_leaves:,}\n")
        f.write(f"  - Profundidade Média das Árvores: {avg_depth:.2f}\n")
        f.write(f"  - Profundidade Mínima: {min_depth}\n")
        f.write(f"  - Profundidade Máxima: {max_depth}\n")
        if hasattr(classifier, 'oob_score_') and classifier.oob_score_:
            f.write(f"  - Out-of-Bag Score: {classifier.oob_score_:.4f} ({classifier.oob_score_*100:.2f}%)\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("IMPORTÂNCIA DAS FEATURES\n")
        f.write("=" * 80 + "\n\n")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importância': classifier.feature_importances_
        }).sort_values('Importância', ascending=False)
        
        for idx, row in feature_importance.iterrows():
            f.write(f"{row['Feature']:30s} : {row['Importância']:.6f} ({row['Importância']*100:.2f}%)\n")
    
    print(f"Estrutura da floresta salva: {output_file}")
    return output_file

def main():
    """Função principal para treinar e avaliar o modelo Random Forest"""
    print("Random Forest - Treinamento e Avaliação")
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
    
    class_report, cm, n_estimators, n_features = evaluate_model(classifier, X_test, y_test, y_pred)
    
    # Exporta estrutura da floresta
    export_forest_structure(classifier, feature_columns)
    
    # Calcular estatísticas das árvores para exibição
    total_nodes = sum(tree.tree_.node_count for tree in classifier.estimators_)
    total_leaves = sum(tree.tree_.n_leaves for tree in classifier.estimators_)
    max_depth_trees = [tree.tree_.max_depth for tree in classifier.estimators_]
    avg_depth = sum(max_depth_trees) / len(max_depth_trees)
    
    print("\n" + "=" * 50)
    print("RESULTADOS")
    print("=" * 50)
    print(f"Acurácia Geral: {metrics_dict['accuracy']:.4f} ({metrics_dict['accuracy']*100:.2f}%)")
    if metrics_dict['oob_score']:
        print(f"OOB Score: {metrics_dict['oob_score']:.4f} ({metrics_dict['oob_score']*100:.2f}%)")
    print(f"F1-Score Macro: {metrics_dict['f1_macro']:.4f}")
    print(f"F1-Score Weighted: {metrics_dict['f1_weighted']:.4f}")
    print(f"Precisão Macro: {metrics_dict['precision_macro']:.4f}")
    print(f"Recall Macro: {metrics_dict['recall_macro']:.4f}")
    print(f"Tempo de treinamento: {metrics_dict['training_time']:.2f}s")
    
    if metrics_dict['roc_auc_ovr']:
        print(f"ROC AUC: {metrics_dict['roc_auc_ovr']:.4f}")
    
    print("\n" + "=" * 50)
    print("INFORMAÇÕES DO MODELO")
    print("=" * 50)
    print(f"Número de Árvores: {n_estimators}")
    print(f"Número de Features: {n_features}")
    print(f"Critério: {classifier.criterion}")
    print(f"Total de Nós: {total_nodes:,}")
    print(f"Total de Folhas: {total_leaves:,}")
    print(f"Profundidade Média: {avg_depth:.2f}")
    
    print(f"\nModelo salvo: random_forest_model.joblib")
    
    return metrics_dict['accuracy']

if __name__ == "__main__":
    accuracy = main()
