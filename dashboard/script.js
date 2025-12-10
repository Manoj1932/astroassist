async function sendCommand() {
    const input = document.getElementById("commandInput");
    const text = input.value;

    const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const data = await res.json();

    document.getElementById("result").innerText = 
        "Predicted Intent: " + data.intent;

    const li = document.createElement("li");
    li.innerText = text + " ➝ " + data.intent;
    document.getElementById("history").appendChild(li);

    input.value = "";
}
