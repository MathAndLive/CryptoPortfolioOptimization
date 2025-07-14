import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import re


class HybridNewsAnalyzer:
    """
    Класс для гибридного анализа тональности новостей с использованием FinBERT
    и кастомной логики для оценки влияния, категории и достоверности.
    """

    def __init__(self, impact_multipliers, low_certainty_keywords):
        print("Инициализация анализатора...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)

        self.impact_multipliers = impact_multipliers
        self.low_certainty_keywords = low_certainty_keywords

        print(f"Анализатор готов. Используемое устройство: {self.device.upper()}")

    def _get_finbert_score(self, text: str) -> float:
        """Анализирует текст с помощью FinBERT и возвращает единую оценку от -1 до 1."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prob_positive, prob_negative, _ = probs[0][0].item(), probs[0][1].item(), probs[0][2].item()

        score = prob_positive - prob_negative
        return score

    def _get_category_and_multiplier(self, text: str) -> tuple[str, float]:
        """Определяет категорию новости и возвращает множитель влияния."""
        text_lower = text.lower()
        for category, data in self.impact_multipliers.items():
            # Используем поиск по целым словам, чтобы избежать ложных срабатываний
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', text_lower) for keyword in data['keywords']):
                return category, data['multiplier']
        return 'General', 1.0

    def _get_certainty_factor(self, text: str) -> float:
        """Определяет фактор достоверности на основе ключевых слов."""
        text_lower = text.lower()
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', text_lower) for keyword in self.low_certainty_keywords):
            return 0.65
        return 1.0

    def _handle_negations(self, text: str, base_score: float) -> float:
        """Инвертирует оценку, если находит прямое отрицание негативного события."""
        text_lower = text.lower()
        negation_patterns = [r'not\s(a\s)?(scam|hack|exploit|theft)', r'will\snot\sdelist']
        if any(re.search(pattern, text_lower) for pattern in negation_patterns):
            # Если найдено отрицание, и оценка была негативной, делаем ее слабо-позитивной
            if base_score < -0.1:
                return 0.15
        return base_score

    def analyze(self, text: str) -> dict:
        """Проводит полный гибридный анализ и возвращает детализированный результат."""
        finbert_score = self._get_finbert_score(text)
        category, impact_multiplier = self._get_category_and_multiplier(text)
        certainty_factor = self._get_certainty_factor(text)

        # Применяем логику отрицаний
        final_base_score = self._handle_negations(text, finbert_score)

        final_score = final_base_score * impact_multiplier * certainty_factor
        final_score_clipped = np.clip(final_score, -1.0, 1.0)

        return {
            "text": text,
            "finbert_score": finbert_score,
            "category": category,
            "final_score": final_score_clipped
        }


# --- ВЕСА И КОНФИГУРАЦИЯ ---

IMPACT_MULTIPLIERS = {
    'Security & Risk': {
        'keywords': ['hack', 'vulnerability', 'attack', 'scam', 'fraud', 'theft', 'risk', 'stolen', 'exploit'],
        'multiplier': 1.5},
    'Regulation & Legal': {
        'keywords': ['sec', 'lawsuit', 'sues', 'regulation', 'bill', 'ban', 'sanctions', 'government', 'legal'],
        'multiplier': 1.4},
    'Major Adoption': {
        'keywords': ['etf', 'blackrock', 'goldman', 'jpmorgan', 'nasdaq', 'mastercard', 'visa', 'returns'],
        'multiplier': 1.3},
    'General Adoption & Tech': {
        'keywords': ['partners', 'integrates', 'adopts', 'launches', 'enables', 'platform', 'raises', 'surges'],
        'multiplier': 1.1},
    'Market & Speculation': {'keywords': ['predicts', 'forecast', 'analysis', 'whale', 'price', 'bullish', 'bearish'],
                             'multiplier': 0.8}
}

LOW_CERTAINTY_KEYWORDS = ['rumor', 'sources say', 'suggests', 'proposes', 'plans', 'could', 'potential', 'may']

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    analyzer = HybridNewsAnalyzer(IMPACT_MULTIPLIERS, LOW_CERTAINTY_KEYWORDS)

    news_to_analyze = [
        "The plaintiffs claim that Dolic and Ebel began to conduct unauthorized business activities when the corporate governance of its parent company, Enigma, collapsed.",
        "Blockchain Security Company CoolBitX Raised $16.75 Million in Series B",
        "Uphold will not delist XRP before court decision",
        "Hacker returns stolen funds from $40M GMX exploit"
    ]

    analysis_results = [analyzer.analyze(news) for news in news_to_analyze]
    df_results = pd.DataFrame(analysis_results)

    print("\n--- Результаты улучшенного анализа ---")
    print(df_results)