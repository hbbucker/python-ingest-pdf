import logging
from search import search_prompt
from config import GOOGLE_CHAT_MODEL, OPENAI_CHAT_MODEL, formatLLMError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

def main():
    print("Inicializando chat...")
    chain = search_prompt()
    print("Pronto! Digite sua pergunta ou 'sair'/'exit' para encerrar.\n")

    while True:
        question = input("Você: ").strip()
        if not question:
            continue
        if question.lower() in ("sair", "exit"):
            print("Encerrando chat.")
            break
        try:
            response = chain.invoke(question)
            print(f"\nAssistente: {response}\n")
        except Exception as e:
            print(f"\nErro: serviço de IA indisponível — {formatLLMError(e)}\n")

if __name__ == "__main__":
    main()
