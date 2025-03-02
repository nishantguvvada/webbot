import { RiRobot2Fill } from "react-icons/ri";

export const Footer = () => {
    return (
        <footer id="footer" className="border-t bg-white mt-auto">
                <div className="container mx-auto px-4 py-8">
                    <div className="grid grid-cols-4 gap-8 mb-8">
                        <div className="space-y-4">
                            <h3 className="text-lg">Product</h3>
                            <ul className="space-y-2">
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Features</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Pricing</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">API</span></li>
                            </ul>
                        </div>
                        <div className="space-y-4">
                            <h3 className="text-lg">Company</h3>
                            <ul className="space-y-2">
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">About</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Blog</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Careers</span></li>
                            </ul>
                        </div>
                        <div className="space-y-4">
                            <h3 className="text-lg">Resources</h3>
                            <ul className="space-y-2">
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Documentation</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Help Center</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Guides</span></li>
                            </ul>
                        </div>
                        <div className="space-y-4">
                            <h3 className="text-lg">Legal</h3>
                            <ul className="space-y-2">
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Privacy</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Terms</span></li>
                                <li><span className="text-neutral-600 hover:text-neutral-900 cursor-pointer">Security</span></li>
                            </ul>
                        </div>
                    </div>
                    <div className="border-t pt-8 flex justify-between items-center">
                        <div className="flex items-center space-x-2">
                            <RiRobot2Fill size={30} />
                            <span className="text-xl">Web Doc Bot</span>
                        </div>
                        <div className="flex space-x-4">
                            <a href="#" className="text-neutral-600 hover:text-neutral-900">
                                <i className="fa-brands fa-twitter text-xl"></i>
                            </a>
                            <a href="#" className="text-neutral-600 hover:text-neutral-900">
                                <i className="fa-brands fa-github text-xl"></i>
                            </a>
                            <a href="#" className="text-neutral-600 hover:text-neutral-900">
                                <i className="fa-brands fa-linkedin text-xl"></i>
                            </a>
                        </div>
                        <p className="text-neutral-600 text-sm">© 2025 Web Doc Bot. All rights reserved.</p>
                    </div>
                </div>
            </footer>
    )
}