import gradio as gr
import joblib
import numpy as np
import pandas as pd
import hdbscan

catboost_model = joblib.load("CatBoostRegressor_model.pkl")
cat_lower = joblib.load("CatBoost_lower.pkl")
cat_upper = joblib.load("CatBoost_upper.pkl")

rf_pipeline = joblib.load("RandomForestRegressor_model.pkl")  
sgd_pipeline = joblib.load("SGDRegressor_model.pkl")     
sgd_bagging = joblib.load("SGD_BaggingInterval.pkl")     

hdbscan_model = joblib.load("hdbscan_model.pkl")
cat_options = joblib.load("category_options.pkl")


def predict_price(room_count, lat, lon, series, material, floor, total_floors, total_area, heating, condition):
    if not (42.800 <= lat <= 42.950 and 74.500 <= lon <= 74.750):
        return """
        <div class="error-card">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
            <div class="error-content">
                <h4>Ошибка в координатах</h4>
                <p>Координаты должны быть в пределах г. Бишкек (широта: 42.800–42.950, долгота: 74.500–74.750).</p>
            </div>
        </div>
        """

    error_messages = []
    if not room_count or room_count < 1:
        error_messages.append("<li>Количество комнат должно быть не менее 1.</li>")
    if not total_area or total_area < 1:
        error_messages.append("<li>Площадь квартиры должна быть больше 0 м².</li>")
    if floor and total_floors and floor > total_floors:
        error_messages.append("<li>Этаж не может быть выше общего количества этажей.</li>")

    if error_messages:
        return f"""
        <div class="error-card">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
            <div class="error-content">
                <h4>Обнаружены ошибки</h4>
                <ul>{"".join(error_messages)}</ul>
            </div>
        </div>
        """
        
    coords = np.array([[lat, lon]])
    label, _ = hdbscan.approximate_predict(hdbscan_model, coords)
    cluster = str(label[0])

    input_data = [[
        float(room_count), float(lat), float(lon), series, material,
        float(floor), float(total_floors), float(total_area), heating, condition, cluster
    ]]

    columns = [
        'room_count', 'lat', 'lon', 'Серия', 'house_material',
        'floor', 'total_floors', 'total_area', 'Отопление', 'Состояние', 'hdbscan_cluster'
    ]

    input_df = pd.DataFrame(input_data, columns=columns)

    pred_cat = catboost_model.predict(input_data)[0]
    lower_cat = cat_lower.predict(input_data)[0]
    upper_cat = cat_upper.predict(input_data)[0]

    rf_model = rf_pipeline.named_steps['rf']
    X_transformed = rf_pipeline.named_steps['preprocess'].transform(input_df)

    preds_rf_all = [tree.predict(X_transformed)[0] for tree in rf_model.estimators_]
    pred_rf = np.mean(preds_rf_all)
    lower_rf = np.percentile(preds_rf_all, 2.5)
    upper_rf = np.percentile(preds_rf_all, 97.5)

    preds_sgd_all = [est.predict(input_df)[0] for est in sgd_bagging.estimators_]
    pred_sgd = np.mean(preds_sgd_all)
    lower_sgd = np.percentile(preds_sgd_all, 2.5)
    upper_sgd = np.percentile(preds_sgd_all, 97.5)

    avg_prediction = (pred_cat + pred_rf + pred_sgd) / 3
    
    def format_price(p):
        return f"{p:,.0f}".replace(",", " ")

    results_html = f"""
    <div class="results-container">
        <div class="main-result-card">
            <div class="price-main">{format_price(avg_prediction)} $</div>
        </div>
        
        <h4 class="details-header">Детализация по моделям</h4>
        
        <div class="models-grid">
            <div class="model-card">
                <div class="model-header">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a10 10 0 0 0 10-10H2a10 10 0 0 0 10 10Z"/><path d="M2 12a10 10 0 0 1 10-10c5.52 0 10 4.48 10 10"/><path d="m15.5 8.5-3 3-3-3"/><path d="m12.5 11.5-3 3-3-3"/></svg>
                    <h5>CatBoost</h5>
                </div>
                <div class="price-model">{format_price(pred_cat)} $</div>
                <div class="confidence-interval">
                    <span>95% интервал:</span>
                    <strong>{format_price(lower_cat)} $ – {format_price(upper_cat)} $</strong>
                </div>
            </div>
            
            <div class="model-card">
                <div class="model-header">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 5a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"/><path d="M17 5a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"/><path d="M3 21a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"/><path d="M21 21a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"/><path d="M12 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"/><path d="M8 7.5a4.5 4.5 0 0 1 7.21-3.5"/><path d="M3.28 17.5A4.5 4.5 0 0 1 8 13.5"/><path d="M13.5 16a4.5 4.5 0 0 1 4.21-3.5"/><path d="m13.5 8 2 3.5"/><path d="M8.5 16.5 7 14"/></svg>
                    <h5>Random Forest</h5>
                </div>
                <div class="price-model">{format_price(pred_rf)} $</div>
                <div class="confidence-interval">
                    <span>95% интервал:</span>
                    <strong>{format_price(lower_rf)} $ – {format_price(upper_rf)} $</strong>
                </div>
            </div>
            
            <div class="model-card">
                <div class="model-header">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/><path d="m15 5 3 3"/><path d="M14.5 11.5 16 13"/><path d="M7.5 4.5 9 6"/><path d="m14 10 1-1"/><path d="m5 13 1-1"/></svg>
                    <h5>SGD Bagging</h5>
                </div>
                <div class="price-model">{format_price(pred_sgd)} $</div>
                <div class="confidence-interval">
                    <span>95% интервал:</span>
                    <strong>{format_price(lower_sgd)} $ – {format_price(upper_sgd)} $</strong>
                </div>
            </div>
        </div>
    </div>
    """
    
    return results_html

custom_css = """
:root {
    --primary-color: #4f46e5;
    --primary-hover: #4338ca;
    --background-color: #f8fafc;
    --card-background: #ffffff;
    --text-color: #1e293b;
    --text-light: #64748b;
    --border-color: #e2e8f0;
    --radius: 12px;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}
body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.gradio-container { background-color: var(--background-color); }
.main_row { gap: 24px; }
.main-header {
    text-align: center; padding: 32px 16px; background-color: var(--card-background);
    border: 1px solid var(--border-color); border-radius: var(--radius); margin-bottom: 24px;
}
.main-header h1 { font-size: 28px; font-weight: 700; color: var(--text-color); margin: 0 0 8px; }
.main-header p { font-size: 16px; color: var(--text-light); margin: 0; }
.gradio-group {
    background: var(--card-background); padding: 24px !important; border-radius: var(--radius) !important;
    border: 1px solid var(--border-color); box-shadow: var(--shadow); margin-bottom: 20px !important;
}
.gradio-group h3 {
    margin: 0 0 20px 0 !important; font-size: 18px !important; font-weight: 600 !important;
    color: var(--text-color) !important; border-bottom: 1px solid var(--border-color); padding-bottom: 12px;
}
.gradio-button {
    background: var(--primary-color) !important; border: none !important; color: white !important;
    font-size: 16px !important; font-weight: 600 !important; border-radius: 8px !important;
    box-shadow: var(--shadow) !important; transition: background-color 0.2s ease, transform 0.1s ease !important;
}
.gradio-button:hover { background: var(--primary-hover) !important; transform: translateY(-1px); }
.gradio-textbox, .gradio-dropdown {
    border-radius: 8px !important; border: 1px solid #d1d5db !important;
    font-size: 14px !important; background-color: #f9fafb !important;
}
.gradio-textbox:focus, .gradio-dropdown:focus {
    border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
}
.coordinate-hint {
    background-color: #eef2ff; color: #3730a3; padding: 12px; border-radius: 8px;
    margin: -10px 0 15px 0; font-size: 13px; border-left: 3px solid var(--primary-color);
}
.coordinate-hint p { margin: 0; line-height: 1.5; }
.results-container { padding: 8px; }
/* --- ИЗМЕНЕНИЕ: Стили для главной карточки с результатом --- */
.main-result-card {
    background: var(--card-background);
    border: 2px solid var(--primary-color);
    border-radius: var(--radius);
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
}
.price-main {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.2;
    color: var(--primary-color);
}
/* --- Конец изменений --- */
.details-header {
    font-size: 18px; font-weight: 600; color: var(--text-color); margin: 24px 0 16px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border-color);
}
.models-grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
.model-card {
    background: var(--card-background); border: 1px solid var(--border-color);
    border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow);
}
.model-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.model-header svg { color: var(--primary-color); }
.model-header h5 { margin: 0; font-size: 16px; font-weight: 600; color: var(--text-color); }
.price-model { font-size: 24px; font-weight: 700; color: var(--text-color); margin-bottom: 12px; }
.confidence-interval {
    background: #f9fafb; padding: 10px; border-radius: 8px; font-size: 14px; text-align: center;
}
.confidence-interval span { color: var(--text-light); display: block; margin-bottom: 4px; }
.confidence-interval strong { color: var(--text-color); }
.error-card {
    background-color: #fff1f2; color: #be123c; border: 1px solid #fecdd3; padding: 20px;
    border-radius: var(--radius); display: flex; align-items: flex-start; gap: 16px;
}
.error-card svg { flex-shrink: 0; margin-top: 3px; }
.error-content h4 { margin: 0 0 8px 0; font-size: 16px; font-weight: 600; }
.error-content p, .error-content ul { margin: 0; padding-left: 1.2em; font-size: 14px; }
.error-content li { margin-bottom: 4px; }
.placeholder-output {
    display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;
    background: var(--card-background); border: 1px dashed var(--border-color); border-radius: var(--radius);
    padding: 40px; text-align: center; color: var(--text-light);
}
.placeholder-output svg { margin-bottom: 16px; color: var(--border-color); }
.placeholder-output h3 { font-size: 18px; color: var(--text-color); margin: 0 0 8px; }
"""

with gr.Blocks(css=custom_css, title="Оценка квартир в Бишкеке") as demo:
    
    gr.HTML("""
    <div class="main-header">
        <h1>Прогнозирование стоимости квартиры в Бишкеке</h1>
        <p>Используйте модели машинного обучения для получения точной рыночной оценки</p>
    </div>
    """)
    
    with gr.Row(elem_classes="main_row"):
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("<h3>Характеристики квартиры</h3>")
                with gr.Row():
                    room_count = gr.Number(label="Количество комнат")
                    total_area = gr.Number(label="Общая площадь (м²)")
                with gr.Row():
                    floor = gr.Number(label="Этаж")
                    total_floors = gr.Number(label="Всего этажей")

            with gr.Group():
                gr.Markdown("<h3>Местоположение</h3>")
                gr.HTML("""
                <div class="coordinate-hint">
                    <p><strong>Подсказка:</strong> Координаты можно получить в Google Maps, нажав правой кнопкой мыши по нужной локации. Центр Бишкека: 42.875, 74.603</p>
                </div>
                """)
                with gr.Row():
                    lat = gr.Number(label="Широта", value=42.875763)
                    lon = gr.Number(label="Долгота", value=74.603676)
            
            with gr.Group():
                gr.Markdown("<h3>Характеристики дома</h3>")
                with gr.Row():
                    series = gr.Dropdown(sorted(cat_options["Серия"]), label="Серия дома", value=sorted(cat_options["Серия"])[0] if cat_options["Серия"] else None)
                    material = gr.Dropdown(cat_options["house_material"], label="Материал дома", value=cat_options["house_material"][0] if cat_options["house_material"] else None)
                with gr.Row():
                    heating = gr.Dropdown(cat_options["Отопление"], label="Отопление", value=cat_options["Отопление"][0] if cat_options["Отопление"] else None)
                    condition = gr.Dropdown(cat_options["Состояние"], label="Состояние", value=cat_options["Состояние"][0] if cat_options["Состояние"] else None)
            
            btn = gr.Button("Рассчитать стоимость")

        with gr.Column(scale=1):
            output = gr.HTML("""
            <div class="placeholder-output">
                <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><path d="M22 21H3c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2h18c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2zM8 15v-4c0-.55.45-1 1-1h6c.55 0 1 .45 1 1v4H8z"/><path d="M16 3v2"/><path d="M8 3v2"/><path d="M3 7h18"/></svg>
                <h3>Готов к оценке</h3>
                <p>Заполните все поля слева, чтобы получить результат.</p>
            </div>
            """)

    btn.click(
        predict_price,
        inputs=[room_count, lat, lon, series, material, floor, total_floors, total_area, heating, condition],
        outputs=output
    )

    gr.HTML("""
    <div style="text-align: center; padding: 32px; color: #94a3b8; font-size: 14px;">
        <p>Система оценки работает на ансамбле моделей машинного обучения: CatBoost, Random Forest и SGD.</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()