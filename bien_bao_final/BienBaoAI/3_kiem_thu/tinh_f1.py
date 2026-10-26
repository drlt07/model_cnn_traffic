
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

file = sys.argv[1] if len(sys.argv) > 1 else 'ket_qua.csv'
df = pd.read_csv(file)
df = df.rename(columns={'Nhãn đúng': 'nhan_dung', 'Dự đoán': 'du_doan',
                        'Biển thật (nhãn)': 'nhan_dung', 'App dự đoán (nhãn)': 'du_doan'})
df = df.dropna(subset=['nhan_dung', 'du_doan'])
y, p = df['nhan_dung'].astype(int), df['du_doan'].astype(int)

lops = sorted(set(y))   # chỉ tính các lớp có trong bộ ảnh thử
P, R, F, S = precision_recall_fscore_support(y, p, labels=lops, zero_division=0)

print(f'Số ảnh: {len(df)}   Đúng: {(y == p).sum()}   Accuracy: {accuracy_score(y, p) * 100:.2f}%\n')
print(f'{"Nhãn":>5} {"Số ảnh":>7} {"TP":>4} {"FP":>4} {"FN":>4} {"P":>6} {"R":>6} {"F1":>6}')
for i, lop in enumerate(lops):
    tp = ((y == lop) & (p == lop)).sum()
    fp = ((y != lop) & (p == lop)).sum()
    fn = ((y == lop) & (p != lop)).sum()
    print(f'{lop:>5} {S[i]:>7} {tp:>4} {fp:>4} {fn:>4} {P[i]:>6.2f} {R[i]:>6.2f} {F[i]:>6.2f}')
print(f'\nMacro-Precision = {P.mean():.2f}   Macro-Recall = {R.mean():.2f}   Macro-F1 = {F.mean():.2f}')

bang = pd.DataFrame({'Nhãn': lops, 'Số ảnh': S, 'Precision': P.round(2), 'Recall': R.round(2), 'F1': F.round(2)})
bang.to_csv('bang_f1.csv', index=False, encoding='utf-8-sig')

cm = confusion_matrix(y, p, labels=lops)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(lops)), lops, rotation=90, fontsize=7)
plt.yticks(range(len(lops)), lops, fontsize=7)
plt.xlabel('Nhãn dự đoán')
plt.ylabel('Nhãn đúng')
plt.title('Ma trận nhầm lẫn')
plt.tight_layout()
plt.savefig('ma_tran_nham_lan.png', dpi=200)
print('Đã lưu bang_f1.csv và ma_tran_nham_lan.png')
