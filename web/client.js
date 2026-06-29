var pc = null;
var currentAvatar = null;
var avatarCatalog = [];
var choiceState = {
    initialized: false,
    treeId: "default_choice_tree",
    current: null,
    path: []
};
var ssvepState = {
    enabled: false,
    originalColor: false,
    rafId: null,
    frameCnt: 0,
    actualFps: 60,
    lutLen: 1000,
    defaultFrequencies: [12.8, 11.2, 8.8],
    defaultPhases: [0, 0, 0],
    targets: []
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function(ch) {
        return {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "\"": "&quot;",
            "'": "&#39;"
        }[ch];
    });
}

function buildAvatarPlaceholder(avatar) {
    var title = escapeHtml(avatar.name || avatar.id || "Avatar");
    return "data:image/svg+xml;charset=UTF-8," + encodeURIComponent(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 600'>" +
        "<defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>" +
        "<stop stop-color='#dce8ff' offset='0%'/>" +
        "<stop stop-color='#f7fbff' offset='100%'/>" +
        "</linearGradient></defs>" +
        "<rect width='480' height='600' fill='url(#g)'/>" +
        "<circle cx='240' cy='190' r='58' fill='#1149d8' opacity='0.75'/>" +
        "<rect x='150' y='266' width='180' height='132' rx='58' fill='#1149d8' opacity='0.72'/>" +
        "<text x='240' y='470' font-size='34' text-anchor='middle' fill='#122033' font-family='Segoe UI, Arial, sans-serif' font-weight='700'>" + title + "</text>" +
        "<text x='240' y='512' font-size='20' text-anchor='middle' fill='#66758a' font-family='Segoe UI, Arial, sans-serif'>LiveTalking</text>" +
        "</svg>"
    );
}

function normalizeAvatarCatalog(avatars) {
    return avatars.map(function(avatar, index) {
        var normalized = {
            id: avatar.id,
            name: avatar.name || avatar.id,
            description: avatar.description || ("数字人角色 " + (index + 1)),
            image: avatar.image || null
        };
        normalized.placeholder = buildAvatarPlaceholder(normalized);
        normalized.badge = "角色 " + (index + 1);
        return normalized;
    });
}

function getAvatarById(avatarId) {
    return avatarCatalog.find(function(item) {
        return item.id === avatarId;
    }) || null;
}

function ensureCurrentAvatar() {
    if (!currentAvatar && avatarCatalog.length > 0) {
        currentAvatar = avatarCatalog[0];
    }
}

function updateConnectionStatus(status) {
    var statusIndicator = $("#connection-status");
    var statusText = $("#status-text");

    statusIndicator.removeClass("status-connected status-disconnected status-connecting");

    switch (status) {
        case "connected":
            statusIndicator.addClass("status-connected");
            statusText.text("已连接");
            break;
        case "connecting":
            statusIndicator.addClass("status-connecting");
            statusText.text("连接中...");
            break;
        default:
            statusIndicator.addClass("status-disconnected");
            statusText.text("未连接");
            break;
    }
}

function addChatMessage(message, type) {
    var messageType = type || "user";
    var messageClass = messageType === "user" ? "user-message" : "system-message";
    var sender = messageType === "user" ? "你" : "系统";
    var messageElement = $(
        '<div class="asr-text ' + messageClass + '">' +
            escapeHtml(sender) + "：" + escapeHtml(message) +
        "</div>"
    );

    $("#chat-messages").append(messageElement);
    var container = document.getElementById("chat-messages");
    container.scrollTop = container.scrollHeight;
}

function ensureSessionReady() {
    var sessionid = String(document.getElementById("sessionid").value || "0");
    return sessionid && sessionid !== "0";
}

function postJson(url, body) {
    return fetch(url, {
        body: JSON.stringify(body),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.text().then(function(text) {
            var payload = {};
            if (text) {
                try {
                    payload = JSON.parse(text);
                } catch (error) {
                    throw new Error(url + " 返回的不是 JSON: " + text.slice(0, 240));
                }
            }
            if (!response.ok || payload.code !== 0) {
                throw new Error(payload.msg || (url + " 请求失败: " + response.status));
            }
            return payload;
        });
    });
}

function formatFrequency(value) {
    return Number(value).toFixed(value % 1 === 0 ? 0 : 1) + "Hz";
}

function getChoiceFrequency(choice, index) {
    var ssvep = choice.ssvep || {};
    var configured = ssvep.frequency || choice.ssvep_frequency;
    var frequency = Number(configured);
    if (isFinite(frequency) && frequency > 0) {
        return frequency;
    }
    return ssvepState.defaultFrequencies[index % ssvepState.defaultFrequencies.length];
}

function getChoicePhase(choice) {
    var ssvep = choice.ssvep || {};
    var configured = ssvep.phase || choice.ssvep_phase;
    var phase = Number(configured);
    if (isFinite(phase)) {
        return phase;
    }
    return ssvepState.defaultPhases[0];
}

function buildSsvepLut(frequency, phase, actualFps, lutLen) {
    var lut = [];
    var fps = actualFps || 60;
    for (var i = 0; i < lutLen; i += 1) {
        var intensity = (
            (Math.sin(2 * Math.PI * frequency * i / fps + phase * Math.PI) + 1) / 2
        ) * 0.9 + 0.1;
        lut.push(intensity);
    }
    return lut;
}

function estimateRefreshRate(sampleFrames) {
    sampleFrames = sampleFrames || 90;
    if (!window.requestAnimationFrame) {
        return Promise.resolve(60);
    }

    return new Promise(function(resolve) {
        var times = [];

        function step(now) {
            times.push(now);
            if (times.length < sampleFrames) {
                window.requestAnimationFrame(step);
                return;
            }

            var intervals = [];
            for (var i = 1; i < times.length; i += 1) {
                intervals.push(times[i] - times[i - 1]);
            }
            intervals.sort(function(a, b) {
                return a - b;
            });

            var median = intervals[Math.floor(intervals.length / 2)] || 16.6667;
            var fps = Math.round(1000 / median);
            if (!isFinite(fps) || fps < 30 || fps > 240) {
                fps = 60;
            }
            resolve(fps);
        }

        window.requestAnimationFrame(step);
    });
}

function updateSsvepStatus() {
    var status = $("#ssvep-status");
    if (!status.length) {
        return;
    }

    if (!ssvepState.enabled) {
        status.text("关闭时按普通按钮选择；开启后选项刺激块按 LUT 正弦亮度闪烁。");
        return;
    }

    if (!ssvepState.targets.length) {
        status.text("SSVEP 已开启，等待选项加载。" + (ssvepState.originalColor ? "当前为原色显示。" : ""));
        return;
    }

    status.text("SSVEP 已开启，FPS " + Math.round(ssvepState.actualFps || 60) + "：" + ssvepState.targets.map(function(target, index) {
        return "选项 " + (index + 1) + " " + formatFrequency(target.frequency);
    }).join("，"));
}

function resetSsvepButtonVisuals() {
    $("#choice-options .choice-btn")
        .removeClass("ssvep-locked")
        .css({
            backgroundColor: "",
            borderColor: "",
            boxShadow: "",
            color: ""
        });
    $("#choice-options .ssvep-stimulus").css({
        backgroundColor: "",
        borderColor: ""
    });
}

function stopSsvepFlicker() {
    if (ssvepState.rafId !== null && window.cancelAnimationFrame) {
        window.cancelAnimationFrame(ssvepState.rafId);
        ssvepState.rafId = null;
    }
    $("#choice-options").removeClass("ssvep-enabled ssvep-original-color");
    resetSsvepButtonVisuals();
}

function startSsvepFlicker() {
    stopSsvepFlicker();
    if (!ssvepState.enabled || !ssvepState.targets.length || !window.requestAnimationFrame) {
        updateSsvepStatus();
        return;
    }

    $("#choice-options")
        .addClass("ssvep-enabled")
        .toggleClass("ssvep-original-color", ssvepState.originalColor);
    ssvepState.frameCnt = 0;
    ssvepState.actualFps = ssvepState.actualFps || 60;
    ssvepState.targets.forEach(function(target) {
        target.lut = buildSsvepLut(
            target.frequency,
            target.phase || 0,
            ssvepState.actualFps,
            ssvepState.lutLen
        );
    });

    function tick() {
        ssvepState.targets.forEach(function(target) {
            if (!target.stimulus || !target.lut) {
                return;
            }
            var intensity = target.lut[ssvepState.frameCnt % target.lut.length];
            var value = Math.round(intensity * 255);
            target.stimulus.style.backgroundColor = "rgb(" + value + "," + value + "," + value + ")";
        });
        ssvepState.frameCnt += 1;
        ssvepState.rafId = window.requestAnimationFrame(tick);
    }

    ssvepState.rafId = window.requestAnimationFrame(tick);
    updateSsvepStatus();
}

function setSsvepOriginalColor(enabled) {
    ssvepState.originalColor = Boolean(enabled);
    $("#ssvep-original-color-toggle").prop("checked", ssvepState.originalColor);
    $("#choice-options").toggleClass("ssvep-original-color", ssvepState.enabled && ssvepState.originalColor);
    updateSsvepStatus();
}

function setSsvepEnabled(enabled) {
    ssvepState.enabled = Boolean(enabled);
    $("#ssvep-toggle").prop("checked", ssvepState.enabled);
    if (ssvepState.enabled) {
        startSsvepFlicker();
        estimateRefreshRate(90).then(function(fps) {
            if (!ssvepState.enabled) {
                return;
            }
            ssvepState.actualFps = fps;
            startSsvepFlicker();
        });
    } else {
        stopSsvepFlicker();
        updateSsvepStatus();
    }
}

function selectChoiceBySsvepTarget(targetIndex) {
    var index = Number(targetIndex);
    if (!isFinite(index)) {
        return false;
    }

    if (index >= 1) {
        index -= 1;
    }

    var target = ssvepState.targets[index];
    if (!target) {
        return false;
    }

    $(target.element).addClass("ssvep-locked");
    requestChoiceSelect(target.choiceId);
    return true;
}

function renderChoiceState(payload) {
    stopSsvepFlicker();
    ssvepState.targets = [];

    if (!payload || !payload.current) {
        $("#choice-answer").text("当前还没有可用的选项对话状态。");
        $("#choice-path").text("当前路径：未初始化");
        $("#choice-options").empty();
        updateSsvepStatus();
        return;
    }

    choiceState.initialized = true;
    choiceState.current = payload.current;
    choiceState.path = payload.path || [];
    choiceState.treeId = payload.tree_id || choiceState.treeId;

    $("#choice-answer").text(payload.current.display_text || payload.current.answer_text || "");
    $("#choice-path").text("当前路径：" + (choiceState.path.length ? choiceState.path.join(" > ") : "root"));
    $("#choice-note").text(payload.current.audio_cache_hit ? "当前节点音频已命中缓存，起播会更快。" : "当前节点文本已更新，系统正在尽量提前准备后续音频。");

    var container = $("#choice-options");
    container.empty();
    (payload.current.choices || []).forEach(function(choice, index) {
        var frequency = getChoiceFrequency(choice, index);
        var phase = getChoicePhase(choice);
        var button = $(
            '<button type="button" class="choice-btn ssvep-choice" data-choice-id="' + escapeHtml(choice.choice_id) + '" data-ssvep-index="' + index + '">' +
                '<span class="choice-label">' +
                    '<span class="choice-text">' +
                        '<span class="choice-number">' + (index + 1) + "</span>" +
                        escapeHtml(choice.choice_text) +
                    "</span>" +
                    '<span class="choice-ssvep-side">' +
                        '<span class="ssvep-frequency-badge">' + formatFrequency(frequency) + "</span>" +
                        '<span class="ssvep-stimulus" aria-hidden="true"></span>' +
                    "</span>" +
                "</span>" +
            "</button>"
        );
        container.append(button);
        ssvepState.targets.push({
            index: index,
            choiceId: choice.choice_id,
            frequency: frequency,
            phase: phase,
            element: button[0],
            stimulus: button.find(".ssvep-stimulus")[0]
        });
    });

    if (ssvepState.enabled) {
        startSsvepFlicker();
    } else {
        updateSsvepStatus();
    }
}

function requestChoiceInit() {
    if (!ensureSessionReady()) {
        alert("请先开始连接");
        return Promise.resolve();
    }

    return fetch("/choice/init", {
        body: JSON.stringify({
            sessionid: String(document.getElementById("sessionid").value),
            tree_id: choiceState.treeId
        }),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.json();
    }).then(function(payload) {
        if (payload.code !== 0) {
            throw new Error(payload.msg || "choice init failed");
        }
        renderChoiceState(payload.data);
    }).catch(function(error) {
        console.error(error);
        $("#choice-note").text("选项对话初始化失败：" + error.message);
    });
}

function requestChoiceReset() {
    if (!ensureSessionReady()) {
        alert("请先开始连接");
        return Promise.resolve();
    }

    return fetch("/choice/reset", {
        body: JSON.stringify({
            sessionid: String(document.getElementById("sessionid").value)
        }),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.json();
    }).then(function(payload) {
        if (payload.code !== 0) {
            throw new Error(payload.msg || "choice reset failed");
        }
        renderChoiceState(payload.data);
    }).catch(function(error) {
        console.error(error);
        $("#choice-note").text("重新开始失败：" + error.message);
    });
}

function requestChoiceSelect(choiceId) {
    if (!ensureSessionReady()) {
        alert("请先开始连接");
        return;
    }

    var selectedButton = $('#choice-options .choice-btn[data-choice-id="' + choiceId + '"]');
    var selectedLabel = selectedButton.find(".choice-text").text().trim() || selectedButton.text().trim();
    $("#choice-options .choice-btn").prop("disabled", true);
    fetch("/choice/select", {
        body: JSON.stringify({
            sessionid: String(document.getElementById("sessionid").value),
            choice_id: choiceId,
            interrupt: true
        }),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.json();
    }).then(function(payload) {
        if (payload.code !== 0) {
            throw new Error(payload.msg || "choice select failed");
        }
        renderChoiceState(payload.data);
        if (selectedLabel) {
            addChatMessage("选择了：" + selectedLabel, "user");
        }
    }).catch(function(error) {
        console.error(error);
        $("#choice-note").text("选项切换失败：" + error.message);
    }).finally(function() {
        $("#choice-options .choice-btn").prop("disabled", false);
    });
}

function switchPage(page) {
    $(".page-panel").removeClass("active");
    if (page === "conversation") {
        $("#conversation-page").addClass("active");
    } else {
        $("#avatar-selection-page").addClass("active");
    }
}

function bindImageFallback($images) {
    $images.each(function() {
        var image = this;
        image.onerror = function() {
            var fallback = image.getAttribute("data-fallback");
            if (fallback && image.src !== fallback) {
                image.src = fallback;
            }
        };
    });
}

function updateSelectedAvatarDisplay() {
    ensureCurrentAvatar();
    if (!currentAvatar) {
        return;
    }

    $("#selection-note").text("当前选择：" + currentAvatar.name);
    $("#selected-avatar-name").text(currentAvatar.name);
    $("#selected-avatar-description").text(currentAvatar.description);
    $("#side-avatar-name").text(currentAvatar.name);
    $("#selected-avatar-image")
        .attr("src", currentAvatar.image || currentAvatar.placeholder)
        .attr("data-fallback", currentAvatar.placeholder)
        .attr("alt", currentAvatar.name);
    bindImageFallback($("#selected-avatar-image"));
    $("#chat-welcome-message").text("系统：当前角色为“" + currentAvatar.name + "”，点击“开始连接”后即可开始对话。");
}

function setCurrentAvatar(avatarId) {
    var nextAvatar = getAvatarById(avatarId);
    if (!nextAvatar) {
        return;
    }

    currentAvatar = nextAvatar;
    $(".avatar-card").removeClass("selected");
    $('.avatar-card[data-avatar-id="' + currentAvatar.id + '"]').addClass("selected");
    $("#enter-chat-btn").prop("disabled", false);
    updateSelectedAvatarDisplay();
}

function renderAvatarCards() {
    var container = $("#avatar-grid");
    container.empty();

    avatarCatalog.forEach(function(avatar) {
        var imageSource = avatar.image || avatar.placeholder;
        var card = $(
            '<button class="avatar-card text-start" type="button" data-avatar-id="' + escapeHtml(avatar.id) + '">' +
                '<div class="avatar-preview">' +
                    '<img src="' + escapeHtml(imageSource) + '" alt="' + escapeHtml(avatar.name) + '" data-fallback="' + escapeHtml(avatar.placeholder) + '">' +
                    '<span class="avatar-check"><i class="bi bi-check-lg"></i></span>' +
                "</div>" +
                '<div class="avatar-card-body">' +
                    '<div class="avatar-card-title">' +
                        "<h3>" + escapeHtml(avatar.name) + "</h3>" +
                        '<span class="avatar-tag">' + escapeHtml(avatar.badge) + "</span>" +
                    "</div>" +
                    "<p>" + escapeHtml(avatar.description) + "</p>" +
                "</div>" +
            "</button>"
        );
        container.append(card);
    });

    bindImageFallback(container.find("img[data-fallback]"));
    if (avatarCatalog.length > 0) {
        setCurrentAvatar(avatarCatalog[0].id);
    }
}

function loadAvatarCatalog() {
    return fetch("/api/avatars")
        .then(function(response) {
            return response.text().then(function(text) {
                var payload;
                try {
                    payload = JSON.parse(text);
                } catch (error) {
                    throw new Error("角色列表接口返回的不是 JSON: " + text.slice(0, 240));
                }

                if (!response.ok || payload.code !== 0) {
                    throw new Error(payload.msg || ("角色列表接口请求失败: " + response.status));
                }

                return payload.data || {};
            });
        })
        .then(function(data) {
            var avatars = Array.isArray(data.avatars) ? data.avatars : [];
            avatarCatalog = normalizeAvatarCatalog(avatars);
            renderAvatarCards();

            if (avatarCatalog.length === 0) {
                $("#selection-note").text("未扫描到可用角色");
            }
        })
        .catch(function(error) {
            console.error(error);
            $("#selection-note").text("角色列表加载失败");
            addChatMessage("角色列表加载失败：" + error.message, "system");
        });
}

function negotiate() {
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });

    return pc.createOffer()
        .then(function(offer) {
            return pc.setLocalDescription(offer);
        })
        .then(function() {
            return new Promise(function(resolve) {
                if (pc.iceGatheringState === "complete") {
                    resolve();
                } else {
                    var checkState = function() {
                        if (pc.iceGatheringState === "complete") {
                            pc.removeEventListener("icegatheringstatechange", checkState);
                            resolve();
                        }
                    };
                    pc.addEventListener("icegatheringstatechange", checkState);
                }
            });
        })
        .then(function() {
            var offer = pc.localDescription;
            return fetch("/offer", {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                    avatar: currentAvatar ? currentAvatar.id : undefined
                }),
                headers: {
                    "Content-Type": "application/json"
                },
                method: "POST"
            });
        })
        .then(function(response) {
            return response.text().then(function(text) {
                var payload;
                try {
                    payload = JSON.parse(text);
                } catch (error) {
                    throw new Error("/offer 返回的不是 JSON: " + text.slice(0, 240));
                }

                if (!response.ok) {
                    throw new Error(payload.msg || ("创建会话失败: " + response.status));
                }

                if (!payload.sdp || !payload.type || !payload.sessionid) {
                    throw new Error("/offer 返回缺少必要字段: " + text.slice(0, 240));
                }

                return payload;
            });
        })
        .then(function(answer) {
            document.getElementById("sessionid").value = answer.sessionid;
            return pc.setRemoteDescription(answer).then(function() {
                return requestChoiceInit();
            });
        })
        .catch(function(error) {
            updateConnectionStatus("disconnected");
            $("#stop").hide();
            $("#start").show();
            alert(error.message || String(error));
        });
}

function start() {
    if (!currentAvatar) {
        alert("请先选择一个角色");
        return;
    }

    var config = {
        sdpSemantics: "unified-plan"
    };

    if (document.getElementById("use-stun").checked) {
        config.iceServers = [{ urls: ["stun:stun.l.google.com:19302"] }];
    }

    pc = new RTCPeerConnection(config);

    pc.addEventListener("track", function(evt) {
        if (evt.track.kind === "video") {
            document.getElementById("video").srcObject = evt.streams[0];
        } else {
            document.getElementById("audio").srcObject = evt.streams[0];
        }
    });

    pc.addEventListener("connectionstatechange", function() {
        if (!pc) {
            return;
        }
        if (pc.connectionState === "connected") {
            updateConnectionStatus("connected");
        } else if (pc.connectionState === "connecting") {
            updateConnectionStatus("connecting");
        } else if (["failed", "closed", "disconnected"].indexOf(pc.connectionState) >= 0) {
            updateConnectionStatus("disconnected");
        }
    });

    document.getElementById("start").style.display = "none";
    document.getElementById("stop").style.display = "inline-block";
    updateConnectionStatus("connecting");
    negotiate();
}

function stop() {
    stopSsvepFlicker();
    ssvepState.targets = [];
    document.getElementById("stop").style.display = "none";
    document.getElementById("start").style.display = "inline-block";
    document.getElementById("sessionid").value = "0";
    updateConnectionStatus("disconnected");
    choiceState.initialized = false;
    choiceState.current = null;
    choiceState.path = [];
    $("#choice-answer").text("点击“开始连接”后，可以在这里使用三选一的引导式对话。");
    $("#choice-path").text("当前路径：未初始化");
    $("#choice-options").empty();
    $("#choice-note").text("系统会优先返回文本和选项，并在后台尽量提前准备下一轮音频。");
    updateSsvepStatus();

    if (pc) {
        setTimeout(function() {
            if (pc) {
                pc.close();
                pc = null;
            }
        }, 300);
    }
}

window.onunload = function() {
    setTimeout(function() {
        if (pc) {
            pc.close();
        }
    }, 300);
};

window.onbeforeunload = function(e) {
    setTimeout(function() {
        if (pc) {
            pc.close();
        }
    }, 300);
    e = e || window.event;
    if (e) {
        e.returnValue = "确定离开当前页面吗？";
    }
    return "确定离开当前页面吗？";
};

$(document).ready(function() {
    updateConnectionStatus("disconnected");
    loadAvatarCatalog();
    updateSsvepStatus();

    window.LiveTalkingSSVEP = {
        enable: function() {
            setSsvepEnabled(true);
        },
        disable: function() {
            setSsvepEnabled(false);
        },
        setOriginalColor: setSsvepOriginalColor,
        selectTarget: selectChoiceBySsvepTarget
    };

    $("#avatar-grid").on("click", ".avatar-card", function() {
        setCurrentAvatar($(this).data("avatarId"));
    });

    $("#enter-chat-btn").on("click", function() {
        if (!currentAvatar) {
            return;
        }
        switchPage("conversation");
    });

    $("#back-to-selection-btn").on("click", function() {
        stop();
        switchPage("selection");
    });

    $("#video-size-slider").on("input", function() {
        var value = $(this).val();
        $("#video-size-value").text(value + "%");
        $("#video").css("width", value + "%");
    });

    $("#start").on("click", function() {
        start();
    });

    $("#stop").on("click", function() {
        stop();
    });

    $("#choice-init-btn").on("click", function() {
        requestChoiceInit();
    });

    $("#choice-reset-btn").on("click", function() {
        requestChoiceReset();
    });

    $("#ssvep-toggle").on("change", function() {
        setSsvepEnabled(this.checked);
    });

    $("#ssvep-original-color-toggle").on("change", function() {
        setSsvepOriginalColor(this.checked);
    });

    $("#choice-options").on("click", ".choice-btn", function() {
        requestChoiceSelect($(this).data("choiceId"));
    });

    $("#btn_start_record").on("click", function() {
        fetch("/record", {
            body: JSON.stringify({
                type: "start_record",
                sessionid: String(document.getElementById("sessionid").value)
            }),
            headers: {
                "Content-Type": "application/json"
            },
            method: "POST"
        }).then(function(response) {
            if (response.ok) {
                $("#btn_start_record").prop("disabled", true);
                $("#btn_stop_record").prop("disabled", false);
                $("#recording-indicator").addClass("active");
            }
        }).catch(function(error) {
            console.error("Error:", error);
        });
    });

    $("#btn_stop_record").on("click", function() {
        fetch("/record", {
            body: JSON.stringify({
                type: "end_record",
                sessionid: String(document.getElementById("sessionid").value)
            }),
            headers: {
                "Content-Type": "application/json"
            },
            method: "POST"
        }).then(function(response) {
            if (response.ok) {
                $("#btn_start_record").prop("disabled", false);
                $("#btn_stop_record").prop("disabled", true);
                $("#recording-indicator").removeClass("active");
            }
        }).catch(function(error) {
            console.error("Error:", error);
        });
    });

    $("#echo-form").on("submit", function(e) {
        e.preventDefault();
        var message = $("#message").val();
        if (!message.trim()) {
            return;
        }

        postJson("/human", {
                text: message,
                type: "echo",
                interrupt: true,
                sessionid: String(document.getElementById("sessionid").value)
        }).then(function() {
            addChatMessage('已发送播报文本 "' + message + '"', "system");
        }).catch(function(error) {
            console.error(error);
            addChatMessage("播报请求失败：" + error.message, "system");
        });

        $("#message").val("");
    });

    $("#chat-form").on("submit", function(e) {
        e.preventDefault();
        var message = $("#chat-message").val();
        if (!message.trim()) {
            return;
        }

        postJson("/human", {
                text: message,
                type: "chat",
                interrupt: true,
                sessionid: String(document.getElementById("sessionid").value)
        }).catch(function(error) {
            console.error(error);
            addChatMessage("聊天请求失败：" + error.message, "system");
        });

        addChatMessage(message, "user");
        $("#chat-message").val("");
    });

    var mediaRecorder = null;
    var isRecording = false;
    var recognition = null;
    var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "zh-CN";

        recognition.onresult = function(event) {
            var interimTranscript = "";
            var finalTranscript = "";

            for (var i = event.resultIndex; i < event.results.length; i += 1) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                    $("#chat-message").val(interimTranscript);
                }
            }

            if (finalTranscript) {
                $("#chat-message").val(finalTranscript);
            }
        };
    }

    $("#voice-record-btn").on("mousedown touchstart", function(e) {
        e.preventDefault();
        startVoiceRecording();
    }).on("mouseup mouseleave touchend", function() {
        if (isRecording) {
            stopVoiceRecording();
        }
    });

    function startVoiceRecording() {
        if (isRecording) {
            return;
        }

        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            isRecording = true;
            $("#voice-record-btn").addClass("recording-pulse").css("background-color", "#dc2626");

            if (recognition) {
                recognition.start();
            }
        }).catch(function(error) {
            console.error("Microphone access failed:", error);
            alert("无法获取麦克风权限，请检查浏览器设置。");
        });
    }

    function stopVoiceRecording() {
        if (!isRecording || !mediaRecorder) {
            return;
        }

        mediaRecorder.stop();
        isRecording = false;
        mediaRecorder.stream.getTracks().forEach(function(track) {
            track.stop();
        });

        $("#voice-record-btn").removeClass("recording-pulse").css("background-color", "");

        if (recognition) {
            recognition.stop();
        }

        setTimeout(function() {
            var recognizedText = $("#chat-message").val().trim();
            if (!recognizedText) {
                return;
            }

            postJson("/human", {
                    text: recognizedText,
                    type: "chat",
                    interrupt: true,
                    sessionid: String(document.getElementById("sessionid").value)
            }).catch(function(error) {
                console.error(error);
                addChatMessage("语音发送失败：" + error.message, "system");
            });

            addChatMessage(recognizedText, "user");
            $("#chat-message").val("");
        }, 500);
    }
});
