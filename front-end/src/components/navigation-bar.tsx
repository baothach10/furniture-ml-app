import {
    Navbar,
    NavbarBrand,
    NavbarContent,
    NavbarItem,
    NavbarMenuToggle,
    NavbarMenu,
    NavbarMenuItem
} from "@nextui-org/navbar";
import { Button } from "@nextui-org/react";
import Link from "next/link";

export default function NavigationBar() {
    return (
        <Navbar>
            <NavbarBrand>
                <Link href="/">
                    <p className="font-bold text-red-600">Group 01</p>
                </Link>
            </NavbarBrand>
            <NavbarContent className="flex justify-between" justify="center">
                <NavbarItem>
                    <Link href="/model/task1">
                        Task 1
                    </Link>
                </NavbarItem>
                <NavbarItem >
                    <Link href="/model/task2">
                        Task 2
                    </Link>
                </NavbarItem>
                <NavbarItem>
                    <Link href="/model/task3">
                        Task 3
                    </Link>
                </NavbarItem>
            </NavbarContent>
        </Navbar>
    )
}