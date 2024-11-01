"use client";

import { useState, useEffect, useRef } from "react";
import { Send, MapPin, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

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
      process.env.NEXT_PUBLIC_WEBSOCKET_URL || "wss://echo.websocket.org";
    const socket = new WebSocket(WEBSOCKET_URL);
    setWs(socket);

    socket.onopen = () => {
      setMessages([
        {
          id: Date.now(),
          content:
            "Hello! I'm your travel assistant. How can I help you plan your trip to Singapore?",
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
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4">
      <Card className="w-full max-w-4xl h-[800px] bg-white rounded-xl shadow-lg overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <h1 className="text-xl font-semibold">Travel Assistant</h1>
          <Badge variant="secondary" className="bg-green-100 text-green-700">
            Online
          </Badge>
        </div>

        <ScrollArea className="flex-1 p-6" ref={scrollAreaRef}>
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    message.sender === "user"
                      ? "bg-black text-white"
                      : "bg-gray-100 text-gray-900"
                  }`}
                >
                  {message.sender === "bot" && message.content.startsWith("{")
                    ? formatJsonResponse(message.content)
                    : message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-900 rounded-lg px-4 py-2 flex items-center space-x-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        <div className="border-t bg-gray-50/50">
          <div className="px-6 py-4">
            <div className="mb-4">
              <p className="text-sm font-medium text-gray-500 mb-2">
                Sample Questions:
              </p>
              <div className="flex flex-wrap gap-2">
                {sampleQuestions.map((question, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    onClick={() => sendMessage(question)}
                    className="text-sm bg-white"
                    disabled={isLoading}
                  >
                    <MapPin className="w-4 h-4 mr-2" />
                    {question}
                  </Button>
                ))}
              </div>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                sendMessage(inputMessage);
              }}
              className="flex gap-2"
            >
              <Input
                type="text"
                placeholder="Type your travel question or choose from the samples above..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                className="flex-grow bg-white"
                disabled={isLoading}
              />
              <Button
                type="submit"
                className="bg-black text-white hover:bg-gray-800 px-6"
                disabled={isLoading}
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <>
                    <Send className="h-4 w-4 mr-2" />
                    Send
                  </>
                )}
              </Button>
            </form>
          </div>
        </div>
      </Card>
    </div>
  );
}
