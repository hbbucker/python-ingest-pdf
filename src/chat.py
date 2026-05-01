from search import search_prompt

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
        response = chain.invoke(question)
        print(f"\nAssistente: {response}\n")

if __name__ == "__main__":
    main()
