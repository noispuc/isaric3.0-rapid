import pandas as pd
import isaric.datacleaning.orchestrator as dc

# Exemplo de DataFrame com duplicatas
data = {
    "patient_id": [101, 102, 103, 101, 104, 102],
    "age": [34, 45, 23, 34, 52, 45],
    "gender": ["M", "F", "F", "M", "M", "F"]
}

df = pd.DataFrame(data)

print("📊 Original DataFrame:")
print(df)

# Aplicar função de remoção de duplicatas
df_clean = dc.remove_duplicates(df)

print("\n✅ DataFrame após remoção de duplicatas:")
print(df_clean)
