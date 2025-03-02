"use client"
import { IoIosSend } from "react-icons/io";
import { RiRobot2Fill } from "react-icons/ri";
import axios from "axios";
import toast from "react-hot-toast";
import { useEffect, useState } from "react";

export const Chat = () => {
    const [userInput, setUserInput] = useState("");
    const [collectionList, setCollectionList] = useState([]);
    const [collectionName, setCollectionName] = useState("");
    const [aiResponse, setAIResponse] = useState("");

    useEffect(() => {
        const listCollection = async () => {
            try {
                const response = await axios.get(`${process.env.NEXT_PUBLIC_URL}/collection`);
                setCollectionList(response.data.response)
            } 
            catch(err) {
                console.log(err)
            }
        }
        listCollection();
    }, []);

    const chatHandler = async () => {
        if (!userInput) {
            toast.error("No user input provided.")
            return
        }

        if (!collectionName) {
            toast.error("Vector Store not selected");
            return
        }
    
        try {

            const response = await axios.post(`${process.env.NEXT_PUBLIC_URL}/response`, {
                user_input: userInput,
                collection_name: collectionName.split('-')[0]
            });

            setAIResponse(response.data.response)
        } catch(err) {
            console.log(err);
            toast.error("Something went wrong!")
        }
    }
    return (
        <div id="chat-interface" className="max-w-3xl mx-auto bg-white rounded-lg shadow-sm border">
            <div className="border-b px-6 py-3 flex justify-between items-center">
                <h2>Chat with Website</h2>
                <select value={collectionName || ""} onChange={(e) => { setCollectionName(e.target.value) }} className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48">
                    <option value="" disabled hidden>Select a Vector Store</option>
                    {collectionList.map((collection, i) => {
                        return <option key={i} value={collection}>{collection}</option>
                    })}
                </select>
            </div>
            <div id="chat-messages" className="h-[400px] overflow-y-auto p-6 space-y-4">
                <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-neutral-200 flex items-center justify-center">
                        <RiRobot2Fill size={20} />
                    </div>
                    <div className="flex-1 bg-neutral-100 rounded-lg p-4">
                        <p className="text-neutral-700">Hello! I'm ready to answer questions about your website. Please paste a URL above and build the vector store to get started.</p>
                        
                    </div>
                </div>
                <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-neutral-200 flex items-center justify-center">
                        <RiRobot2Fill size={20} />
                    </div>
                    <div className="flex-1 bg-neutral-100 rounded-lg p-4">
                        <p className="text-neutral-700">{aiResponse}</p>
                        
                    </div>
                </div>
            </div>
            <div id="chat-input" className="p-4 border-t">
                <div className="flex gap-2">
                    <input onChange={(e) => { setUserInput(e.target.value) }} type="text" placeholder="Ask a question about the website..." className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-neutral-200 focus:outline-none"/>
                    <button onClick={chatHandler} className="px-4 py-2 bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 cursor-pointer">
                        <IoIosSend size={30}/>
                    </button>
                </div>
            </div>
        </div>
    )
}