"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
self["webpackHotUpdate_N_E"]("app/model/task1/page",{

/***/ "(app-pages-browser)/./src/app/model/task1/page.tsx":
/*!**************************************!*\
  !*** ./src/app/model/task1/page.tsx ***!
  \**************************************/
/***/ (function(module, __webpack_exports__, __webpack_require__) {

eval(__webpack_require__.ts("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"(app-pages-browser)/./node_modules/next/dist/compiled/react/jsx-dev-runtime.js\");\n/* harmony import */ var _app_actions_submit_images__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @/app/actions/submit-images */ \"(app-pages-browser)/./src/app/actions/submit-images.ts\");\n/* harmony import */ var next_image__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/image */ \"(app-pages-browser)/./node_modules/next/dist/api/image.js\");\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ \"(app-pages-browser)/./node_modules/next/dist/compiled/react/index.js\");\n/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);\n/* __next_internal_client_entry_do_not_use__ default auto */ \nvar _s = $RefreshSig$();\n\n\n\nconst FileUploadForm = ()=>{\n    _s();\n    const [selectedFile, setSelectedFile] = (0,react__WEBPACK_IMPORTED_MODULE_3__.useState)(null);\n    const [imageUrl, setImageUrl] = (0,react__WEBPACK_IMPORTED_MODULE_3__.useState)(null);\n    const [loading, setLoading] = (0,react__WEBPACK_IMPORTED_MODULE_3__.useState)(false);\n    const [recommendations, setRecommendations] = (0,react__WEBPACK_IMPORTED_MODULE_3__.useState)();\n    const handleFileChange = (event)=>{\n        setSelectedFile(event.target.files[0]);\n        setImageUrl(URL.createObjectURL(event.target.files[0]));\n    };\n    const handleSubmit = async (e)=>{\n        e.preventDefault();\n        setLoading(true);\n        if (selectedFile) {\n            const response = await (0,_app_actions_submit_images__WEBPACK_IMPORTED_MODULE_1__.handleImageSubmit)(selectedFile);\n            setLoading(false);\n            setRecommendations(response.recommendations);\n            console.log(response.recommendations);\n        }\n    };\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n        children: [\n            \"loading: \",\n            loading,\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"h2\", {\n                children: \"File UplhandleSubmitoad\"\n            }, void 0, false, {\n                fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n                lineNumber: 34,\n                columnNumber: 7\n            }, undefined),\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"form\", {\n                onSubmit: handleSubmit,\n                children: [\n                    /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"input\", {\n                        type: \"file\",\n                        onChange: handleFileChange\n                    }, void 0, false, {\n                        fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n                        lineNumber: 36,\n                        columnNumber: 9\n                    }, undefined),\n                    !loading && /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"button\", {\n                        type: \"submit\",\n                        children: \"Upload\"\n                    }, void 0, false, {\n                        fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n                        lineNumber: 37,\n                        columnNumber: 22\n                    }, undefined)\n                ]\n            }, void 0, true, {\n                fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n                lineNumber: 35,\n                columnNumber: 7\n            }, undefined),\n            imageUrl && /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(next_image__WEBPACK_IMPORTED_MODULE_2__[\"default\"], {\n                width: 200,\n                height: 200,\n                src: imageUrl,\n                alt: \"input_image\"\n            }, void 0, false, {\n                fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n                lineNumber: 41,\n                columnNumber: 11\n            }, undefined),\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"div\", {\n                className: \"flex flex-row\",\n                children: recommendations && recommendations.map((rec, index)=>/*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(next_image__WEBPACK_IMPORTED_MODULE_2__[\"default\"], {\n                        width: 200,\n                        height: 200,\n                        src: \"http://localhost:8001/\" + rec,\n                        alt: \"recomendations\"\n                    }, index, false, {\n                        fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n                        lineNumber: 46,\n                        columnNumber: 11\n                    }, undefined))\n            }, void 0, false, {\n                fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n                lineNumber: 44,\n                columnNumber: 7\n            }, undefined)\n        ]\n    }, void 0, true, {\n        fileName: \"/Users/thachngo/Documents/RMIT_DOCUMENTS/LEARNING_PROGRAMS/COSC2753 - Machine Learning/Assignment2/Web UI/furniture-ml-app/front-end/src/app/model/task1/page.tsx\",\n        lineNumber: 32,\n        columnNumber: 5\n    }, undefined);\n};\n_s(FileUploadForm, \"EoTORUQPwfUbJmFfp1P3wtU3Lz8=\");\n_c = FileUploadForm;\n/* harmony default export */ __webpack_exports__[\"default\"] = (FileUploadForm);\nvar _c;\n$RefreshReg$(_c, \"FileUploadForm\");\n\n\n;\n    // Wrapped in an IIFE to avoid polluting the global scope\n    ;\n    (function () {\n        var _a, _b;\n        // Legacy CSS implementations will `eval` browser code in a Node.js context\n        // to extract CSS. For backwards compatibility, we need to check we're in a\n        // browser context before continuing.\n        if (typeof self !== 'undefined' &&\n            // AMP / No-JS mode does not inject these helpers:\n            '$RefreshHelpers$' in self) {\n            // @ts-ignore __webpack_module__ is global\n            var currentExports = module.exports;\n            // @ts-ignore __webpack_module__ is global\n            var prevSignature = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevSignature) !== null && _b !== void 0 ? _b : null;\n            // This cannot happen in MainTemplate because the exports mismatch between\n            // templating and execution.\n            self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.id);\n            // A module can be accepted automatically based on its exports, e.g. when\n            // it is a Refresh Boundary.\n            if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {\n                // Save the previous exports signature on update so we can compare the boundary\n                // signatures. We avoid saving exports themselves since it causes memory leaks (https://github.com/vercel/next.js/pull/53797)\n                module.hot.dispose(function (data) {\n                    data.prevSignature =\n                        self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports);\n                });\n                // Unconditionally accept an update to this module, we'll check if it's\n                // still a Refresh Boundary later.\n                // @ts-ignore importMeta is replaced in the loader\n                module.hot.accept();\n                // This field is set when the previous version of this module was a\n                // Refresh Boundary, letting us know we need to check for invalidation or\n                // enqueue an update.\n                if (prevSignature !== null) {\n                    // A boundary can become ineligible if its exports are incompatible\n                    // with the previous exports.\n                    //\n                    // For example, if you add/remove/change exports, we'll want to\n                    // re-execute the importing modules, and force those components to\n                    // re-render. Similarly, if you convert a class component to a\n                    // function, we want to invalidate the boundary.\n                    if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevSignature, self.$RefreshHelpers$.getRefreshBoundarySignature(currentExports))) {\n                        module.hot.invalidate();\n                    }\n                    else {\n                        self.$RefreshHelpers$.scheduleUpdate();\n                    }\n                }\n            }\n            else {\n                // Since we just executed the code for the module, it's possible that the\n                // new exports made it ineligible for being a boundary.\n                // We only care about the case when we were _previously_ a boundary,\n                // because we already accepted this update (accidental side effect).\n                var isNoLongerABoundary = prevSignature !== null;\n                if (isNoLongerABoundary) {\n                    module.hot.invalidate();\n                }\n            }\n        }\n    })();\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKGFwcC1wYWdlcy1icm93c2VyKS8uL3NyYy9hcHAvbW9kZWwvdGFzazEvcGFnZS50c3giLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7QUFDZ0U7QUFDakM7QUFFUztBQUV4QyxNQUFNSSxpQkFBaUI7O0lBQ3JCLE1BQU0sQ0FBQ0MsY0FBY0MsZ0JBQWdCLEdBQUdILCtDQUFRQSxDQUFjO0lBQzlELE1BQU0sQ0FBQ0ksVUFBVUMsWUFBWSxHQUFHTCwrQ0FBUUEsQ0FBQztJQUN6QyxNQUFNLENBQUNNLFNBQVNDLFdBQVcsR0FBR1AsK0NBQVFBLENBQVU7SUFDaEQsTUFBTSxDQUFDUSxpQkFBaUJDLG1CQUFtQixHQUFHVCwrQ0FBUUE7SUFHdEQsTUFBTVUsbUJBQW1CLENBQUNDO1FBQ3hCUixnQkFBZ0JRLE1BQU1DLE1BQU0sQ0FBQ0MsS0FBSyxDQUFDLEVBQUU7UUFDckNSLFlBQVlTLElBQUlDLGVBQWUsQ0FBQ0osTUFBTUMsTUFBTSxDQUFDQyxLQUFLLENBQUMsRUFBRTtJQUN2RDtJQUdBLE1BQU1HLGVBQWUsT0FBT0M7UUFDMUJBLEVBQUVDLGNBQWM7UUFDaEJYLFdBQVc7UUFDWCxJQUFJTCxjQUFjO1lBQ2hCLE1BQU1pQixXQUFXLE1BQU10Qiw2RUFBaUJBLENBQUNLO1lBQ3pDSyxXQUFXO1lBQ1hFLG1CQUFtQlUsU0FBU1gsZUFBZTtZQUMzQ1ksUUFBUUMsR0FBRyxDQUFDRixTQUFTWCxlQUFlO1FBQ3RDO0lBQ0Y7SUFFQSxxQkFDRSw4REFBQ2M7O1lBQUk7WUFDT2hCOzBCQUNWLDhEQUFDaUI7MEJBQUc7Ozs7OzswQkFDSiw4REFBQ0M7Z0JBQUtDLFVBQVVUOztrQ0FDZCw4REFBQ1U7d0JBQU1DLE1BQUs7d0JBQU9DLFVBQVVsQjs7Ozs7O29CQUM1QixDQUFDSix5QkFBVyw4REFBQ3VCO3dCQUFPRixNQUFLO2tDQUFTOzs7Ozs7Ozs7Ozs7WUFHbkN2QiwwQkFDRSw4REFBQ04sa0RBQUtBO2dCQUFDZ0MsT0FBTztnQkFBS0MsUUFBUTtnQkFBS0MsS0FBSzVCO2dCQUFVNkIsS0FBSTs7Ozs7OzBCQUd2RCw4REFBQ1g7Z0JBQUlZLFdBQVU7MEJBQ1oxQixtQkFBbUJBLGdCQUFnQjJCLEdBQUcsQ0FBQyxDQUFDQyxLQUFLQyxzQkFDNUMsOERBQUN2QyxrREFBS0E7d0JBQUNnQyxPQUFPO3dCQUFLQyxRQUFRO3dCQUFLQyxLQUFLLDJCQUEyQkk7d0JBQUtILEtBQUk7dUJBQXNCSTs7Ozs7Ozs7Ozs7Ozs7OztBQUt6RztHQTVDTXBDO0tBQUFBO0FBOENOLCtEQUFlQSxjQUFjQSxFQUFDIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vX05fRS8uL3NyYy9hcHAvbW9kZWwvdGFzazEvcGFnZS50c3g/MWZmZiJdLCJzb3VyY2VzQ29udGVudCI6WyJcInVzZSBjbGllbnRcIlxuaW1wb3J0IHsgaGFuZGxlSW1hZ2VTdWJtaXQgfSBmcm9tICdAL2FwcC9hY3Rpb25zL3N1Ym1pdC1pbWFnZXMnO1xuaW1wb3J0IEltYWdlIGZyb20gJ25leHQvaW1hZ2UnO1xuXG5pbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XG5cbmNvbnN0IEZpbGVVcGxvYWRGb3JtID0gKCkgPT4ge1xuICBjb25zdCBbc2VsZWN0ZWRGaWxlLCBzZXRTZWxlY3RlZEZpbGVdID0gdXNlU3RhdGU8RmlsZSB8IG51bGw+KG51bGwpO1xuICBjb25zdCBbaW1hZ2VVcmwsIHNldEltYWdlVXJsXSA9IHVzZVN0YXRlKG51bGwpXG4gIGNvbnN0IFtsb2FkaW5nLCBzZXRMb2FkaW5nXSA9IHVzZVN0YXRlPGJvb2xlYW4+KGZhbHNlKVxuICBjb25zdCBbcmVjb21tZW5kYXRpb25zLCBzZXRSZWNvbW1lbmRhdGlvbnNdID0gdXNlU3RhdGU8RmlsZVtdPigpXG5cblxuICBjb25zdCBoYW5kbGVGaWxlQ2hhbmdlID0gKGV2ZW50OiB7IHRhcmdldDogeyBmaWxlczogUmVhY3QuU2V0U3RhdGVBY3Rpb248bnVsbD5bXTsgfTsgfSkgPT4ge1xuICAgIHNldFNlbGVjdGVkRmlsZShldmVudC50YXJnZXQuZmlsZXNbMF0pO1xuICAgIHNldEltYWdlVXJsKFVSTC5jcmVhdGVPYmplY3RVUkwoZXZlbnQudGFyZ2V0LmZpbGVzWzBdKSlcbiAgfTtcblxuXG4gIGNvbnN0IGhhbmRsZVN1Ym1pdCA9IGFzeW5jIChlKSA9PiB7XG4gICAgZS5wcmV2ZW50RGVmYXVsdCgpXG4gICAgc2V0TG9hZGluZyh0cnVlKVxuICAgIGlmIChzZWxlY3RlZEZpbGUpIHtcbiAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgaGFuZGxlSW1hZ2VTdWJtaXQoc2VsZWN0ZWRGaWxlKTtcbiAgICAgIHNldExvYWRpbmcoZmFsc2UpXG4gICAgICBzZXRSZWNvbW1lbmRhdGlvbnMocmVzcG9uc2UucmVjb21tZW5kYXRpb25zKVxuICAgICAgY29uc29sZS5sb2cocmVzcG9uc2UucmVjb21tZW5kYXRpb25zKVxuICAgIH1cbiAgfVxuXG4gIHJldHVybiAoXG4gICAgPGRpdj5cbiAgICAgIGxvYWRpbmc6IHtsb2FkaW5nfVxuICAgICAgPGgyPkZpbGUgVXBsaGFuZGxlU3VibWl0b2FkPC9oMj5cbiAgICAgIDxmb3JtIG9uU3VibWl0PXtoYW5kbGVTdWJtaXR9PlxuICAgICAgICA8aW5wdXQgdHlwZT1cImZpbGVcIiBvbkNoYW5nZT17aGFuZGxlRmlsZUNoYW5nZX0gLz5cbiAgICAgICAgeyFsb2FkaW5nICYmIDxidXR0b24gdHlwZT1cInN1Ym1pdFwiPlVwbG9hZDwvYnV0dG9uPn1cbiAgICAgIDwvZm9ybT5cbiAgICAgIHtcbiAgICAgICAgaW1hZ2VVcmwgJiYgKFxuICAgICAgICAgIDxJbWFnZSB3aWR0aD17MjAwfSBoZWlnaHQ9ezIwMH0gc3JjPXtpbWFnZVVybH0gYWx0PSdpbnB1dF9pbWFnZScgLz5cbiAgICAgICAgKVxuICAgICAgfVxuICAgICAgPGRpdiBjbGFzc05hbWU9J2ZsZXggZmxleC1yb3cnPlxuICAgICAgICB7cmVjb21tZW5kYXRpb25zICYmIHJlY29tbWVuZGF0aW9ucy5tYXAoKHJlYywgaW5kZXgpID0+IChcbiAgICAgICAgICA8SW1hZ2Ugd2lkdGg9ezIwMH0gaGVpZ2h0PXsyMDB9IHNyYz17J2h0dHA6Ly9sb2NhbGhvc3Q6ODAwMS8nICsgcmVjfSBhbHQ9J3JlY29tZW5kYXRpb25zJyBrZXk9e2luZGV4fS8+XG4gICAgICAgICkpIH1cbiAgICAgIDwvZGl2PlxuICAgIDwvZGl2ID5cbiAgKTtcbn07XG5cbmV4cG9ydCBkZWZhdWx0IEZpbGVVcGxvYWRGb3JtO1xuXG4iXSwibmFtZXMiOlsiaGFuZGxlSW1hZ2VTdWJtaXQiLCJJbWFnZSIsIlJlYWN0IiwidXNlU3RhdGUiLCJGaWxlVXBsb2FkRm9ybSIsInNlbGVjdGVkRmlsZSIsInNldFNlbGVjdGVkRmlsZSIsImltYWdlVXJsIiwic2V0SW1hZ2VVcmwiLCJsb2FkaW5nIiwic2V0TG9hZGluZyIsInJlY29tbWVuZGF0aW9ucyIsInNldFJlY29tbWVuZGF0aW9ucyIsImhhbmRsZUZpbGVDaGFuZ2UiLCJldmVudCIsInRhcmdldCIsImZpbGVzIiwiVVJMIiwiY3JlYXRlT2JqZWN0VVJMIiwiaGFuZGxlU3VibWl0IiwiZSIsInByZXZlbnREZWZhdWx0IiwicmVzcG9uc2UiLCJjb25zb2xlIiwibG9nIiwiZGl2IiwiaDIiLCJmb3JtIiwib25TdWJtaXQiLCJpbnB1dCIsInR5cGUiLCJvbkNoYW5nZSIsImJ1dHRvbiIsIndpZHRoIiwiaGVpZ2h0Iiwic3JjIiwiYWx0IiwiY2xhc3NOYW1lIiwibWFwIiwicmVjIiwiaW5kZXgiXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///(app-pages-browser)/./src/app/model/task1/page.tsx\n"));

/***/ })

});