from pathlib import Path
p = Path(r'D:\LiLei\Python_projects\PINNSE\src\models\GAKF\train_gakf.py').resolve()
print('path:', p)
for i, parent in enumerate(p.parents):
    print(i, parent)
