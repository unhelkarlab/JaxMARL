import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

labeled_path = 'results_labeled_large/progress.csv'
unlabeled_path = 'results_unlabeled/progress.csv'

sns.set_theme(context='notebook', style='darkgrid')


def plot_true_action_prob(df_path, is_labeled):
    df = pd.read_csv(df_path)
    plt.figure(figsize=(13.7, 10))
    sns.lineplot(df['bc/prob_true_act'])
    plt.title('Model Accuracy', fontsize=42)
    plt.legend(fontsize=38)
    plt.xlabel('Batch number', fontsize=38)
    plt.xticks(fontsize=38)
    if not is_labeled:
        plt.xticks(range(0, 20, 3))
    plt.ylabel('Value', fontsize=38)
    plt.yticks(fontsize=38)
    if is_labeled:
        plt.savefig('acc_la.pdf', format='pdf')
    else:
        plt.savefig('acc_un.pdf', format='pdf')
    # plt.show()


plot_true_action_prob(unlabeled_path, False)
plot_true_action_prob(labeled_path, True)


def plot_losses(df_path, is_labeled):
    df = pd.read_csv(df_path)
    df = df.reset_index()
    melted_df = pd.melt(df,
                        id_vars=['index'],
                        value_vars=['bc/test_loss', 'bc/training_loss'],
                        var_name='Variable',
                        value_name='Values')

    # Plotting with Seaborn
    plt.figure(figsize=(13, 10))
    sns.lineplot(data=melted_df, x='index', y='Values', hue='Variable')
    plt.title('Training and Validation Loss', fontsize=42)
    plt.legend(fontsize=38)
    plt.xlabel('Batch number', fontsize=38)
    plt.xticks(fontsize=38)
    if not is_labeled:
        plt.xticks(range(0, 20, 3))
    plt.ylabel('Loss', fontsize=38)
    plt.yticks(fontsize=38)
    if is_labeled:
        plt.savefig('loss_la.pdf', format='pdf')
    else:
        plt.savefig('loss_un.pdf', format='pdf')
    # plt.show()


plot_losses(unlabeled_path, False)
plot_losses(labeled_path, True)
