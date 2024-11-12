"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Icon from "@/components/ui/icon";
import LogoImg from "@/app/assets/agoda/logo.svg";
import BotImg from "@/app/assets/agoda/bot.png";

type Message = {
  id: number;
  content: string;
  sender: "user" | "bot";
};

const sampleQuestions = [
  "What activities can I do in Singapore?",
  "Tell me more about Singapore's culture",
  "What's the best time to visit Singapore?",
  "Recommend some local food in Singapore",
];

export default function Component() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const WEBSOCKET_URL =
      process.env.NEXT_PUBLIC_WEBSOCKET_URL || "wss://travel-chatbot.grayground-1ee6f428.southeastasia.azurecontainerapps.io/chat";
    const socket = new WebSocket(WEBSOCKET_URL);
    setWs(socket);

    socket.onopen = () => {
      setMessages([
        {
          id: Date.now(),
          content:
            "Hello! I'm your AI travel agent. How can I help you plan your trip to Singapore?",
          sender: "bot",
        },
      ]);
    };

    socket.onmessage = (event) => {
      const newMessage: Message = {
        id: Date.now(),
        content: event.data,
        sender: "bot",
      };
      setMessages((prevMessages) => [...prevMessages, newMessage]);
      setIsLoading(false);
    };

    return () => {
      socket.close();
    };
  }, []);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const sendMessage = (message: string) => {
    if (message.trim() && ws) {
      const newMessage: Message = {
        id: Date.now(),
        content: message,
        sender: "user",
      };
      setMessages((prevMessages) => [...prevMessages, newMessage]);
      ws.send(message);
      setInputMessage("");
      setIsLoading(true);
    }
  };

  const formatValue = (value: any): React.ReactNode => {
    if (Array.isArray(value)) {
      return (
        <div className="ml-4">
          {value.map((item, index) => (
            <div key={index} className="mt-2">
              {typeof item === "object" ? (
                formatJsonResponse(item)
              ) : (
                <span>{item}</span>
              )}
            </div>
          ))}
        </div>
      );
    }

    if (typeof value === "object" && value !== null) {
      return formatJsonResponse(value);
    }

    return <span>{String(value)}</span>;
  };

  const formatJsonResponse = (content: any) => {
    try {
      const jsonData =
        typeof content === "string" ? JSON.parse(content) : content;

      return (
        <div className="space-y-3">
          {Object.entries(jsonData).map(([key, value]) => (
            <div key={key} className="text-sm">
              <span className="font-semibold capitalize">
                {key.replace(/_/g, " ")}:
              </span>{" "}
              {formatValue(value)}
            </div>
          ))}
        </div>
      );
    } catch (error) {
      return <span>{content}</span>;
    }
  };

  return (
    <div className="flex items-start justify-center min-h-screen">
      <div className="overflow-hidden flex flex-col flex-1 h-screen">
        <div className="flex items-center justify-between px-4 py-4">
          <div className="flex items-center">
            <img src={LogoImg} className="logo" alt="AI Travel Agent" />
            <h1 className="text-l pl-4 font-semibold">AI Travel Agent</h1>
          </div>
          <Icon name="info-circle" size="20" className="text-primary" />
        </div>

        <div className="space-y-4 p-4 flex-1 overflow-y-auto h-full">
          {messages.map((message) => (
            <React.Fragment key={message.id}>
              {message.sender === "user" ? (
                <div className="flex gap-2 justify-end">
                  <div className="max-w-[80%]">
                    <div className="rounded-lg text-sm p-4 theme-message-bg-user">
                      {message.content}
                    </div>
                    <div className="text-xs text-gray-500 mt-1 text-right">
                      {new Date(message.id).toLocaleTimeString()}
                    </div>
                  </div>
                  <div className="theme-avatar-user relative inline-flex items-center justify-center w-8 h-8 overflow-hidden rounded-full">
                    <span className="font-medium">JL</span>
                  </div>
                </div>
              ) : (
                <div className="flex gap-2 justify-start">
                  <img src={BotImg.src} alt="Bot" className="w-8 h-8 rounded-full" />
                  <div className="max-w-[80%]">
                    <div className="rounded-lg text-sm p-4 theme-message-bg-bot">
                      {message.content.startsWith("{")
                        ? formatJsonResponse(message.content)
                        : message.content}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {new Date(message.id).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              )}
            </React.Fragment>
          ))}
          {isLoading && (
            <div className="flex gap-2 justify-start">
              <img src={BotImg.src} alt="Bot" className="w-8 h-8 rounded-full" />
              <div className="max-w-[80%]">
                <div className="flex items-center rounded-lg text-sm p-4 theme-message-bg-bot">
                  <Icon name="loader" className="bx-spin mr-2" />
                  Thinking...
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="border-t bg-gray-50/50">
          <div className="flex flex-row items-end gap-2 p-4">
            <div className="flex-1">
              <h2 className="flex items-center text-l font-semibold mb-4">
                <Icon name="message-square-dots mr-2" />
                Quick Questions
              </h2>
              <div className="flex flex-col gap-2">
                {sampleQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => sendMessage(question)}
                    className="text-sm bg-white"
                    disabled={isLoading}
                  >
                    {question}
                  </Button>
                ))}
              </div>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  sendMessage(inputMessage);
                }}
                className="flex gap-2 mt-4"
              >

                <div className="relative w-full">
                  <Input
                    type="text"
                    placeholder="Ask me anything or select above..."
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    className="flex-grow bg-white"
                    disabled={isLoading}
                  />
                  <Button
                    size="icon"
                    variant="ghost"
                    disabled={isLoading}
                    className="absolute inset-y-0 end-0 text-primary"
                  > <Icon name="microphone" size="18" /></Button>
                </div>
              </form>
            </div>
            <div>
              <Button
                type="submit"
                size="icon"
                disabled={isLoading}
              >
                {isLoading ? (
                  <Icon name="loader" className="bx-spin" />
                ) : (
                  <>
                    <Icon name="send" size="18" />
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
