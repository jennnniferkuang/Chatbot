// Start Speech-to-Text
document.getElementById("sttBtn").addEventListener("click", function() {
    document.getElementById("btnContainer").style.display = "none";
    document.getElementById("sttSection").style.display = "block";
    document.getElementById("goBackBtn").style.display = "inline-block";
});

document.getElementById("startBtn").addEventListener("click", function() {
    fetch('/start_stt', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById("sttResult").innerText = data.text;
            runChat(data.text);  // Trigger the chat function
        } else {
            document.getElementById("sttResult").innerText = 'Error: ' + data.message;
        }
    });
});

// Start Text-to-Speech
document.getElementById("ttsBtn").addEventListener("click", function() {
    document.getElementById("btnContainer").style.display = "none";
    document.getElementById("ttsSection").style.display = "block";
    document.getElementById("goBackBtn").style.display = "inline-block";
});

function textToSpeech() {
    const text = document.getElementById("ttsInput").value;
    fetch('/text_to_speech', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Text to speech completed');
        } else {
            alert('Error in text-to-speech: ' + data.message);
        }
    });
}

// Handle Chat
function runChat(userInput) {
    fetch('/run_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        const assistantResponse = data.response;
        alert('ChatGPT: ' + assistantResponse);
        textToSpeech(assistantResponse);  // Convert ChatGPT response to speech
    });
}

// Go Back to the initial screen
function goBack() {
    document.getElementById("btnContainer").style.display = "flex";
    document.getElementById("ttsSection").style.display = "none";
    document.getElementById("sttSection").style.display = "none";
    document.getElementById("goBackBtn").style.display = "none";
}
