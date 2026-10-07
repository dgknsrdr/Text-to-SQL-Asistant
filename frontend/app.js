const questionInput =
    document.getElementById("question");

const sendButton =
    document.getElementById("sendButton");

const messages =
    document.getElementById("messages");


questionInput.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Enter"
            &&
            !event.shiftKey
        ) {

            event.preventDefault();

            askQuestion();
        }
    }
);


function exampleQuestion(question) {

    questionInput.value = question;

    questionInput.focus();
}


function addMessage(
    text,
    type,
    sql = null
) {

    const wrapper =
        document.createElement("div");

    wrapper.classList.add(
        "message"
    );

    if (type === "user") {

        wrapper.classList.add(
            "user-message"
        );

    } else {

        wrapper.classList.add(
            "ai-message"
        );
    }


    const title =
        document.createElement("div");

    title.classList.add(
        "message-title"
    );

    title.textContent =
        type === "user"
            ? "Siz"
            : "Enterprise AI";


    const content =
        document.createElement("div");

    content.classList.add(
        "message-text"
    );

    content.textContent =
        text;


    wrapper.appendChild(
        title
    );

    wrapper.appendChild(
        content
    );


    if (sql) {

        const sqlBox =
            document.createElement("div");

        sqlBox.classList.add(
            "sql-box"
        );

        sqlBox.textContent =
            sql;

        wrapper.appendChild(
            sqlBox
        );
    }


    messages.appendChild(
        wrapper
    );


    messages.scrollTop =
        messages.scrollHeight;
}


async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {
        return;
    }


    addMessage(
        question,
        "user"
    );


    questionInput.value = "";

    sendButton.disabled = true;

    sendButton.textContent =
        "Düşünüyor...";


    try {

        const response =
            await fetch(
                "/api/ask",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        soru: question
                    })
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            addMessage(
                "Hata: "
                + data.error,
                "ai"
            );

            return;
        }


        addMessage(
            data.answer,
            "ai",
            data.sql
        );


    } catch (error) {

        addMessage(
            "Sunucuya bağlanırken hata oluştu.",
            "ai"
        );

        console.error(
            error
        );

    } finally {

        sendButton.disabled = false;

        sendButton.textContent =
            "Gönder";

        questionInput.focus();
    }
}