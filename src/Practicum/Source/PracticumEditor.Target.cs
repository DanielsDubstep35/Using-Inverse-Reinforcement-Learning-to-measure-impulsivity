using UnrealBuildTool;
using System.Collections.Generic;

public class PracticumEditorTarget : TargetRules
{
    public PracticumEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;

        ExtraModuleNames.Add("Practicum");
    }
}