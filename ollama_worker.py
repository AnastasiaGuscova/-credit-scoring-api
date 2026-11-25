
import sys
import json
import requests


OLLAMA_URL = "http://localhost:11434/api/chat"


def send_result(data):
    """
    В stdout отправляется только один JSON-объект.
    Это важно для корректной работы llm_explainer.py.
    """
    sys.stdout.write(
        json.dumps(data, ensure_ascii=False)
    )
    sys.stdout.flush()


def main():
    try:
        payload = json.load(sys.stdin)

        generated_text = []

        thinking_length = 0
        eval_count = None
        done_reason = None

        print(
            "Worker: отправка запроса в Ollama.",
            file=sys.stderr
        )

        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(3, 60)
        ) as response:

            response.raise_for_status()

            # Получаем ответ частями.
            for line in response.iter_lines(
                decode_unicode=True
            ):

                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                try:
                    chunk = json.loads(line)

                except json.JSONDecodeError:
                    print(
                        "Worker: пропущен некорректный JSON-фрагмент.",
                        file=sys.stderr
                    )
                    continue

                message = chunk.get("message", {})

                content = message.get("content", "")

                thinking = message.get("thinking", "")

                if content:
                    generated_text.append(content)

                if thinking:
                    thinking_length += len(thinking)

                if chunk.get("done", False):
                    eval_count = chunk.get("eval_count")
                    done_reason = chunk.get("done_reason")
                    break

        answer = "".join(generated_text).strip()

        if not answer:
            send_result({
                "success": False,
                "error": (
                    "Ollama вернула пустой ответ. "
                    f"Длина thinking: {thinking_length}. "
                    f"Причина завершения: {done_reason}."
                )
            })

            return

        send_result({
            "success": True,
            "text": answer,
            "eval_count": eval_count,
            "done_reason": done_reason
        })

    except requests.exceptions.Timeout:
        send_result({
            "success": False,
            "error": "Истекло время ожидания ответа Ollama."
        })

    except requests.exceptions.RequestException as error:
        send_result({
            "success": False,
            "error": f"Ошибка соединения с Ollama: {error}"
        })

    except Exception as error:
        send_result({
            "success": False,
            "error": str(error)
        })


if __name__ == "__main__":
    main()
