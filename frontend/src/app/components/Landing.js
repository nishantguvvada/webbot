import { RiRobot2Fill } from "react-icons/ri";
import { FaUserAlt } from "react-icons/fa";

import { Vector } from "./Vector";
import { Chat } from "./Chat";
import { Footer } from "./Footer";

export const Landing = () => {
    return (
        <>
            <div className="bg-neutral-50 flex flex-col">
                <header className="border-b bg-white shadow-sm">
                    <div className="container mx-auto px-4 py-4 flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                            <RiRobot2Fill size={30} />
                            <span className="text-xl">Web Doc Bot</span>
                        </div>
                        <button className="p-2 rounded-full hover:bg-neutral-100">
                            <FaUserAlt size={20} />
                        </button>
                    </div>
                </header>
                <section className="h-96 bg-black text-white flex justify-center items-center">
                    <div className="mx-auto px-4 py-20 text-center">
                        <h1 className="text-5xl mb-4">Agentic RAG for reading websites</h1>
                        <p className="text-xl text-neutral-300">Read blogs, news articles or the whole website!</p>
                    </div>
                </section>
            </div>
            <main id="main-content" className="flex-1 container mx-auto px-4 py-8">
                <div id="input-section" className="max-w-3xl mx-auto space-y-6 mb-8">
                    <Vector/>
                </div>
                <Chat/>
            </main>
            <Footer/>
        </>
    )
}