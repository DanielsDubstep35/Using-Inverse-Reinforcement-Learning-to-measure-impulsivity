using UnrealBuildTool;
using System.Collections.Generic;

public class PracticumTarget : TargetRules
{
    public PracticumTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.Latest;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;

        // This line is critical! It tells the standalone build to include your code module
        ExtraModuleNames.Add("Practicum");
    }
}