"use client"
import { useState } from "react";
import { IoMdBuild } from "react-icons/io";
import axios from "axios";
import toast from "react-hot-toast";
import isUrl from 'is-url'

export const Vector = () => {
    const [url, setUrl] = useState("");
    const [vectorName, setVectorName] = useState("");

    const vectorHandler = async () => {

        if (!url || !isUrl(url)) {
            toast.error("URL is invalid or blank")
            return
        }

        if (!vectorName) {
            toast.error("Vector Store Name is invalid or blank")
            return
        }

        try {
            const response = await axios.post(`${process.env.NEXT_PUBLIC_URL}/vector`, {
                web_url: url,
                index_name: vectorName
            });

            console.log("Response", response);
            toast.success("Vector Store Created!")

        } catch (err) {
            toast.error(err)
        }
    }

    return (
        <>
            <div className="bg-white p-4 h-48 rounded-lg shadow-sm border flex flex-row justify-around items-end">
                <div className="w-[80%] h-full">
                    <label className="block text-sm text-neutral-700 mb-2">Website URL</label>
                    <div className="flex gap-2">
                        <input value={url} onChange={(e) => { setUrl(e.target.value) }} type="url" placeholder="https://example.com" className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-neutral-200 focus:outline-none"/>
                    </div>
                    <label className="block text-sm text-neutral-700 mb-2 mt-4">Vector Store Name</label>
                    <div className="flex gap-2">
                        <input value={vectorName} onChange={(e) => { setVectorName(e.target.value) }} type="text" placeholder="news-blog" className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-neutral-200 focus:outline-none"/>
                    </div>
                </div>
                <button onClick={vectorHandler} className="px-8 py-2 bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 cursor-pointer">
                        <IoMdBuild size={30}/>
                </button>
            </div>
        </>
    )
}