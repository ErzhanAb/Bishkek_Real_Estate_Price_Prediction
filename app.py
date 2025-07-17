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
    error_messages = []
    if room_count is None:
        error_messages.append("<li>Укажите количество комнат.</li>")
    if total_area is None:
        error_messages.append("<li>Укажите общую площадь.</li>")
    if floor is None:
        error_messages.append("<li>Укажите этаж.</li>")
    if total_floors is None:
        error_messages.append("<li>Укажите общее количество этажей.</li>")

    if error_messages:
        return f"""
        <div class="error-container">
            <div class="error-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" x2="12" y1="9" y2="13"></line><line x1="12" x2="12.01" y1="17" y2="17"></line></svg>
            </div>
            <div class="error-text-content">
                <h4 class="error-title">Не все поля заполнены</h4>
                <ul class="error-list">{"".join(error_messages)}</ul>
            </div>
        </div>
        """
        
    if not (1 <= room_count <= 20):
        error_messages.append("<li>Количество комнат должно быть в диапазоне от 1 до 20.</li>")
    if not (1 <= total_area <= 1500):
        error_messages.append("<li>Общая площадь должна быть в диапазоне от 1 до 1500 м².</li>")
    if not (0 <= floor <= 40):
        error_messages.append("<li>Этаж должен быть в диапазоне от 0 до 40.</li>")
    if not (1 <= total_floors <= 40):
        error_messages.append("<li>Общее количество этажей должно быть от 1 до 40.</li>")
    if not (42.800 <= lat <= 42.950 and 74.500 <= lon <= 74.750):
        error_messages.append("<li>Координаты должны быть в пределах г. Бишкек.</li>")
    if floor > total_floors:
        error_messages.append("<li>Этаж не может быть выше общего количества этажей.</li>")
        
    if error_messages:
        return f"""
        <div class="error-container">
            <div class="error-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" x2="12" y1="9" y2="13"></line><line x1="12" x2="12.01" y1="17" y2="17"></line></svg>
            </div>
            <div class="error-text-content">
                <h4 class="error-title">Пожалуйста, исправьте следующие ошибки:</h4>
                <ul class="error-list">{"".join(error_messages)}</ul>
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

    return f"""
    <div class="output-container">
        <div class="result-main">
            <div class="label">Средняя рыночная цена</div>
            <div class="price">{format_price(avg_prediction)} $</div>
        </div>
        <div class="result-grid">
            <div class="result-card">
                <div class="card-header">
                    <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M12 22a10 10 0 0 0 10-10H2a10 10 0 0 0 10 10Z"></path><path d="M2 12a10 10 0 0 1 10-10c5.52 0 10 4.48 10 10"></path><path d="m15.5 8.5-3 3-3-3"></path><path d="m12.5 11.5-3 3-3-3"></path></svg>
                    <h3>CatBoost</h3>
                </div>
                <p class="value">{format_price(pred_cat)} $</p>
                <div class="interval">
                    <span class="label">95% интервал:</span>
                    <span class="interval-value">{format_price(lower_cat)} $ – {format_price(upper_cat)} $</span>
                </div>
            </div>
            <div class="result-card">
                 <div class="card-header">
                    <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M10 5a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"></path><path d="M17 5a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"></path><path d="M3 21a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"></path><path d="M21 21a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"></path><path d="M12 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"></path><path d="M8 7.5a4.5 4.5 0 0 1 7.21-3.5"></path><path d="M3.28 17.5A4.5 4.5 0 0 1 8 13.5"></path><path d="M13.5 16a4.5 4.5 0 0 1 4.21-3.5"></path><path d="m13.5 8 2 3.5"></path><path d="M8.5 16.5 7 14"></path></svg>
                    <h3>Random Forest</h3>
                </div>
                <p class="value">{format_price(pred_rf)} $</p>
                <div class="interval">
                    <span class="label">95% интервал:</span>
                    <span class="interval-value">{format_price(lower_rf)} $ – {format_price(upper_rf)} $</span>
                </div>
            </div>
            <div class="result-card">
                 <div class="card-header">
                    <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"></path><path d="m15 5 3 3"></path><path d="M14.5 11.5 16 13"></path><path d="M7.5 4.5 9 6"></path><path d="m14 10 1-1"></path><path d="m5 13 1-1"></path></svg>
                    <h3>SGD Bagging</h3>
                </div>
                <p class="value">{format_price(pred_sgd)} $</p>
                <div class="interval">
                    <span class="label">95% интервал:</span>
                    <span class="interval-value">{format_price(lower_sgd)} $ – {format_price(upper_sgd)} $</span>
                </div>
            </div>
        </div>
    </div>
    """

custom_css = """
/* --- Стили для вывода результатов --- */
.output-container { display: flex; flex-direction: column; gap: 24px; }
.result-main {
    text-align: center; padding: 24px; background: var(--background-fill-primary);
    border-radius: var(--radius-lg); border: 1px solid var(--color-accent-soft);
}
.result-main .label { color: var(--body-text-color-subdued); font-size: var(--text-lg); }
.result-main .price {
    color: var(--color-accent); font-size: 48px; font-weight: 700;
    line-height: 1.2; margin-top: 8px;
}
.result-grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
.result-card {
    padding: 20px; background: var(--background-fill-secondary);
    border-radius: var(--radius-lg); border: 1px solid var(--border-color-primary);
}
.card-header { display: flex; align-items: center; gap: 12px; color: var(--body-text-color); }
.card-header svg { width: 22px; height: 22px; stroke: var(--color-accent); }
.card-header h3 { margin: 0; font-size: var(--text-lg); font-weight: 600; }
.result-card .value { font-size: var(--text-xxl); font-weight: 600; color: var(--body-text-color); margin: 16px 0; }
.result-card .interval {
    padding: 12px; background: var(--background-fill-primary);
    border-radius: var(--radius-md); text-align: center;
}
.interval .label { font-size: var(--text-sm); color: var(--body-text-color-subdued); display: block; }
.interval .interval-value { font-size: var(--text-md); color: var(--body-text-color); font-weight: 500; }
.placeholder { padding: 40px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; }
.placeholder svg { width: 48px; height: 48px; color: var(--body-text-color-subdued); margin-bottom: 16px; }
.placeholder h3 { font-size: var(--text-xl); font-weight: 600; color: var(--body-text-color); margin: 0; }
.placeholder p { font-size: var(--text-lg); color: var(--body-text-color-subdued); margin-top: 8px; max-width: 400px; }
.error-container {
    display: flex; align-items: flex-start; gap: 16px; padding: 20px;
    background-color: var(--color-error-background); border-radius: var(--radius-lg);
}
.error-icon svg {
    width: 28px; height: 28px; stroke: var(--color-error-text-weight);
    flex-shrink: 0; margin-top: 2px;
}
.error-text-content { text-align: left; }
.error-title {
    font-size: var(--text-lg); font-weight: 600;
    color: var(--color-error-text-weight); margin: 0 0 8px 0;
}
.error-list {
    margin: 0; padding-left: 20px; font-size: var(--text-md);
    color: var(--color-error-text-weight);
}
.error-list li { margin-bottom: 4px; color: var(--color-error-text-weight); }
/* --- ИЗМЕНЕНИЕ: Стиль для ссылки на GitHub --- */
.source-link { text-align: center; margin-top: -10px; margin-bottom: 20px; }
.source-link, .source-link a { color: var(--body-text-color-subdued) !important; font-size: var(--text-md); text-decoration: none; }
.source-link a { font-weight: 600; text-decoration: underline; }
.source-link a:hover { text-decoration-color: var(--color-accent); }
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo, 
    secondary_hue=gr.themes.colors.blue,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

with gr.Blocks(theme=theme, css=custom_css, title="Оценка квартир в Бишкеке") as demo:
    gr.HTML("""
    <div style="text-align: center;">
        <h1>Прогнозирование стоимости квартиры в Бишкеке</h1>
        <p>Используйте модели машинного обучения для получения точной рыночной оценки</p>
        <p class="source-link">
            Исходный код доступен на 
            <a href="https://github.com/ErzhanAb/Bishkek_Real_Estate_Price_Prediction" target="_blank">GitHub</a>.
        </p>
    </div>
    """)
    
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            with gr.Accordion("1. Характеристики квартиры", open=True):
                with gr.Row():
                    room_count = gr.Number(label="Количество комнат")
                    total_area = gr.Number(label="Общая площадь (м²)")
                with gr.Row():
                    floor = gr.Number(label="Этаж")
                    total_floors = gr.Number(label="Всего этажей")
            with gr.Accordion("2. Местоположение", open=True):
                gr.Markdown("> **Подсказка:** Координаты можно получить в Google Maps, нажав правой кнопкой мыши по нужной локации. Центр Бишкека: 42.875, 74.603")
                with gr.Row():
                    lat = gr.Number(label="Широта", value=42.875763)
                    lon = gr.Number(label="Долгота", value=74.603676)
            with gr.Accordion("3. Характеристики дома", open=True):
                with gr.Row():
                    series = gr.Dropdown(sorted(cat_options["Серия"]), label="Серия дома", value=sorted(cat_options["Серия"])[0] if cat_options["Серия"] else None)
                    material = gr.Dropdown(cat_options["house_material"], label="Материал дома", value=cat_options["house_material"][0] if cat_options["house_material"] else None)
                with gr.Row():
                    heating = gr.Dropdown(cat_options["Отопление"], label="Отопление", value=cat_options["Отопление"][0] if cat_options["Отопление"] else None)
                    condition = gr.Dropdown(cat_options["Состояние"], label="Состояние", value=cat_options["Состояние"][0] if cat_options["Состояние"] else None)
            btn = gr.Button("Рассчитать стоимость", variant="primary")
        with gr.Column(scale=1):
            output = gr.HTML("""
            <div class="placeholder">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line><rect x="7" y="14" width="4" height="4"></rect></svg>
                <h3>Готов к оценке</h3>
                <p>Заполните все поля и нажмите кнопку.</p>
            </div>
            """)
    btn.click(
        predict_price,
        inputs=[room_count, lat, lon, series, material, floor, total_floors, total_area, heating, condition],
        outputs=output
    )
    gr.Markdown("<p style='text-align: center; color: var(--body-text-color-subdued);'>Система оценки работает на ансамбле моделей: CatBoost, Random Forest и SGD.</p>")

if __name__ == "__main__":
    demo.launch()
